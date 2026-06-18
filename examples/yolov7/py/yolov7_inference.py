#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import os
import glob
import argparse
import cv2
from pathlib import Path
from amlnn.api import AMLNN

MEAN = np.array([0, 0, 0], dtype=np.float32)
STD  = np.array([255, 255, 255], dtype=np.float32)

# Standard YOLOv7 Anchors
ANCHORS = {
    8:  np.array([[12, 16], [19, 36], [40, 28]], dtype=np.float32),
    16: np.array([[36, 75], [76, 55], [72, 146]], dtype=np.float32),
    32: np.array([[142, 110], [192, 243], [459, 401]], dtype=np.float32)
}

def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: name for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not load class names from '{path}'. Fallback to generic IDs.")
        return {}

class_names = load_class_names("../input/coco_80_names.txt")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, scale, (left, top)

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=0.003789, zp=-128, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0

    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    raw_val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)

    return input_tensor, original_img, scale, pad

def postprocess(outputs, scale, pad, data_format='NCHW', strides=[8, 16, 32], conf_threshold=0.25, iou_threshold=0.45):
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Calculate inverse sigmoid threshold for early stopping
    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))

    for scale_idx, output in enumerate(outputs):
        stride = strides[scale_idx]
        anchors = ANCHORS[stride]

        if data_format == 'NCHW':
            # (1, 255, H, W) → (H*W, 3, 85)
            batch_size, channels, height, width = output.shape
            output_reshaped = output.transpose(0, 2, 3, 1).reshape(-1, 3, 85)
        elif data_format == 'NHWC':
            # (1, H, W, 255) → (H*W, 3, 85)
            batch_size, height, width, channels = output.shape
            output_reshaped = output.reshape(-1, 3, 85)
        else:
            raise ValueError(f"Unsupported data format: {data_format}.")

        # 1. Compare raw objectness logits to inverse sigmoid threshold
        obj_preds = output_reshaped[..., 4]
        valid_mask = obj_preds > inv_thresh

        if not np.any(valid_mask):
            continue

        # 2. Extract ONLY the valid cells and their grid/anchor indices
        valid_cells = output_reshaped[valid_mask]
        grid_indices, anchor_indices = np.where(valid_mask)

        # Apply sigmoid ONLY to the valid elements
        valid_cells_sigmoid = 1.0 / (1.0 + np.exp(-valid_cells))

        valid_tx_ty = valid_cells_sigmoid[:, 0:2]
        valid_tw_th = valid_cells_sigmoid[:, 2:4]
        valid_obj = valid_cells_sigmoid[:, 4]
        valid_cls = valid_cells_sigmoid[:, 5:]

        # 3. Calculate final scores (Objectness * Class Probability)
        class_scores = np.max(valid_cls, axis=1)
        class_ids = np.argmax(valid_cls, axis=1)
        scores = valid_obj * class_scores

        # Secondary filter for combined score
        score_mask = scores > conf_threshold
        if not np.any(score_mask):
            continue

        valid_tx_ty = valid_tx_ty[score_mask]
        valid_tw_th = valid_tw_th[score_mask]
        scores = scores[score_mask]
        class_ids = class_ids[score_mask]
        grid_indices = grid_indices[score_mask]
        anchor_indices = anchor_indices[score_mask]

        # 4. Generate grid coordinates dynamically based on surviving indices
        grid_y = (grid_indices // width).astype(np.float32)
        grid_x = (grid_indices % width).astype(np.float32)
        valid_anchors = anchors[anchor_indices]

        # 5. Bounding Box Decoding Formula
        bx_by = (valid_tx_ty * 2.0 - 0.5 + np.stack([grid_x, grid_y], axis=1)) * stride
        bw_bh = (valid_tw_th * 2.0) ** 2 * valid_anchors

        x1 = bx_by[:, 0] - bw_bh[:, 0] / 2
        y1 = bx_by[:, 1] - bw_bh[:, 1] / 2
        x2 = bx_by[:, 0] + bw_bh[:, 0] / 2
        y2 = bx_by[:, 1] + bw_bh[:, 1] / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_class_ids.append(class_ids)

    # Merge all scales
    if not all_boxes:
        return []

    valid_boxes = np.concatenate(all_boxes, axis=0)
    valid_scores = np.concatenate(all_scores, axis=0)
    valid_class_ids = np.concatenate(all_class_ids, axis=0)

    # Map coordinates back to original image
    pad_x, pad_y = pad
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale
    valid_boxes = np.maximum(valid_boxes, 0)

    # NMS
    nms_indices = cv2.dnn.NMSBoxes(
        valid_boxes.tolist(), valid_scores.tolist(), conf_threshold, iou_threshold
    )

    detections = []
    if len(nms_indices) > 0:
        nms_indices = nms_indices.flatten()
        for idx in nms_indices:
            bx1, by1, bx2, by2 = valid_boxes[idx]
            detections.append({
                'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                'confidence': float(valid_scores[idx]),
                'class_name': class_names.get(int(valid_class_ids[idx]), 'unknown')
            })

    return detections

def get_color(conf):
    import colorsys
    hue = (conf * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    return bgr

def draw_detections(img, detections, save_path):
    result_img = img.copy()
    print(f"    Detected {len(detections)} objects")

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = map(int, det['bbox'])
        confidence = det['confidence']
        class_name = det['class_name']

        print(f"      {i}. {class_name} ({confidence:.2f}) -> [{x1}, {y1}, {x2}, {y2}]")
        color = get_color(confidence)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(save_path, result_img)
    print(f"    Result saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Yolov7 Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    tensor_attr = tensor_info["inputs"][0]
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    for i, img_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(img_path)}")
        print("=" * 60)

        try:
            input_tensor, ori_img, scale, pad = preprocess(
                img_path, new_shape=(640, 640), data_format='NHWC', s=s, zp=zp, tensor_type=tensor_type
            )

            outputs = amlnn.inference(input_tensor)

            detections = postprocess(outputs, scale, pad, data_format='NHWC', conf_threshold=0.4, iou_threshold=0.45)

            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(img_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            draw_detections(ori_img, detections, str(save_path))

        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
        print()
    print("=" * 60)

    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
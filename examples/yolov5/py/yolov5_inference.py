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

# YOLOv5 Anchors for 640x640 resolution
ANCHORS = {
    8:  np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),
    16: np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),
    32: np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32)
}

def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: name for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not load class names from '{path}'.")
        return {}

class_names = load_class_names("../input/coco_80_names.txt")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
    pad_w = (new_shape[1] - new_unpad[0]) / 2
    pad_h = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, scale, (left, top)

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=0.003789, zp=-128, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    # Apply YOLOv5 Standard Normalization
    normalized_img = (rgb_img.astype(np.float32) - MEAN) / STD

    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(val, 0, 255).astype(np.uint8)

    return input_tensor, original_img, scale, pad

def postprocess(outputs, scale, pad, data_format='NHWC', strides=[8, 16, 32], conf_threshold=0.25, iou_threshold=0.45):
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Calculate inverse sigmoid threshold for early stopping
    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))

    for scale_idx, output in enumerate(outputs):
        stride = strides[scale_idx]
        anchors = ANCHORS[stride]

        # Safely drop the batch dimension
        out = np.squeeze(output)

        # Extract dynamic height and width from the tensor
        if data_format == 'NCHW':
            # out is (255, H, W)
            channels, height, width = out.shape
            out_reshaped = out.transpose(1, 2, 0).reshape(-1, 3, 85)
        elif data_format == 'NHWC':
            if out.shape[-1] == 255: # out is (H, W, 255)
                height, width, channels = out.shape
                out_reshaped = out.reshape(-1, 3, 85)
            else:                    # out is (255, H, W)
                channels, height, width = out.shape
                out_reshaped = out.transpose(1, 2, 0).reshape(-1, 3, 85)
        else:
            raise ValueError(f"Unsupported data format: {data_format}.")

        # 1. Compare raw objectness logits to inverse sigmoid threshold
        obj_preds = out_reshaped[..., 4]
        valid_mask = obj_preds > inv_thresh

        if not np.any(valid_mask):
            continue

        # 2. Extract only the valid cells and their grid/anchor indices
        valid_cells = out_reshaped[valid_mask]
        grid_indices, anchor_indices = np.where(valid_mask)

        # Apply sigmoid only to the valid elements
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

        # 4. Generate grid coordinates dynamically based on actual tensor width
        grid_y = (grid_indices // width).astype(np.float32)
        grid_x = (grid_indices % width).astype(np.float32)
        valid_anchors = anchors[anchor_indices]

        # 5. YOLO Bounding Box Decoding Formula
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

    # Map coordinates back to original image scaling & padding
    pad_x, pad_y = pad
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale
    valid_boxes = np.maximum(valid_boxes, 0)

    # 6. EXACT PER-CLASS NMS
    detections = []
    unique_classes = np.unique(valid_class_ids)

    for c in unique_classes:
        class_mask = valid_class_ids == c
        c_boxes = valid_boxes[class_mask]
        c_scores = valid_scores[class_mask]

        # Convert back to XYWH specifically for OpenCV's NMS
        c_widths = c_boxes[:, 2] - c_boxes[:, 0]
        c_heights = c_boxes[:, 3] - c_boxes[:, 1]
        c_boxes_xywh = np.stack([c_boxes[:, 0], c_boxes[:, 1], c_widths, c_heights], axis=1)

        nms_indices = cv2.dnn.NMSBoxes(
            c_boxes_xywh.tolist(), c_scores.tolist(), conf_threshold, iou_threshold
        )

        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            for idx in nms_indices:
                bx1, by1, bx2, by2 = c_boxes[idx]
                detections.append({
                    'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                    'confidence': float(c_scores[idx]),
                    'class_id': int(c),
                    'class_name': class_names.get(int(c), f'class_{int(c)}')
                })

    return detections

def get_class_color(class_id):
    import colorsys
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    return bgr

def draw_detections(img, detections, save_path):
    result_img = img.copy()

    print(f"    Detected {len(detections)} objects")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        confidence = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']

        print(f"      {i+1}. {class_name} ({confidence:.2f}) -> [{x1}, {y1}, {x2}, {y2}]")

        color = get_class_color(class_id)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(save_path, result_img)
    print(f"    Image saved to: {save_path}")
    return result_img

def main():
    parser = argparse.ArgumentParser(description="Yolov5 Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()
    print(amlnn.get_sdk_version())

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files.sort()

    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        amlnn.uninit()
        return

    tensor_attr = tensor_info["inputs"][0]
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])
    
    model_stem = Path(args.model_path).stem
    result_dir = f"{model_stem}_result"
    os.makedirs(result_dir, exist_ok=True)

    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            input_tensor, original_img, scale, pad = preprocess(
                image_path, new_shape=(640, 640), data_format='NHWC', s=s, zp=zp, tensor_type=tensor_type
            )

            outputs = amlnn.inference(inputs=[input_tensor])

            detections = postprocess(outputs, scale, pad, data_format='NHWC', conf_threshold=0.5, iou_threshold=0.45)

            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            draw_detections(original_img, detections, str(save_path))

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        print()
    print(f"=" * 60)
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
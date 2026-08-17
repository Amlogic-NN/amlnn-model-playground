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
import colorsys
from pathlib import Path
from amlnn.api import AMLNN

# YOLOv5 Anchors for 640x640 resolution
ANCHORS = {
    8:  np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),
    16: np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),
    32: np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32)
}

class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'doughnut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def letterbox(img, new_shape, color=(114, 114, 114)):
    shape = img.shape[:2]  # [height, width]

    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    pad_h = new_shape[0] - new_unpad[1]
    pad_w = new_shape[1] - new_unpad[0]

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, scale, (left, top)

def preprocess(img_path, new_shape, s, zp, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    # 1. Resize and pad
    processed_img, scale, pad = letterbox(original_img, new_shape)

    # 2. BGR to RGB
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    rgb_float = rgb_img.astype(np.float32)

    # 3. Fused Normalization & Quantization if needed
    if tensor_type == 0: # FP32 & FP16
        input_tensor = rgb_float / 255.0 # Only normalize
    elif tensor_type in (2, 3, 4):
        inv_scale = np.float32(1.0 / (255.0 * s))
        raw_val = np.round((rgb_float * inv_scale) + zp)

        if tensor_type == 2: # Int8
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3: # Uint8
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else: # Int16
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, original_img, scale, pad

def postprocess(outputs, input_shape, scale, pad, conf_threshold, iou_threshold):
    # input_shape is a tuple: (input_height, input_width)
    input_h, input_w = input_shape

    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Calculate inverse sigmoid threshold for early stopping
    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))

    strides = [8, 16, 32]
    for idx, output in enumerate(outputs):
        # output shape is [batch_size, height, width, channels]
        batch_size, height, width, channels = output.shape

        # 3 anchors, with 4 box values + 1 objectness value + 80 class values per anchor
        num_anchors = 3
        if channels % num_anchors != 0:
            raise ValueError(f"Invalid YOLOv5 channel count: {channels}")

        values_per_anchor = channels // num_anchors
        num_classes = values_per_anchor - 5
        output_reshaped = output.reshape(-1, num_anchors, values_per_anchor)

        # 1. Stride and anchor selection
        stride = strides[idx]
        anchors = ANCHORS[stride]

        expected_height = input_h // stride
        expected_width = input_w // stride
        if height != expected_height or width != expected_width:
            raise ValueError(f"Unexpected output shape {output.shape} for stride {stride}; expected height={expected_height}, width={expected_width}")

        if num_classes <= 0:
            raise ValueError(f"Invalid YOLOv5 channel count: {channels}")

        # 2. Compare raw objectness logits to inverse sigmoid threshold
        obj_preds = output_reshaped[..., 4]
        valid_mask = obj_preds > inv_thresh

        if not np.any(valid_mask):
            continue

        # 3. Extract only the valid cells and their grid/anchor indices
        valid_cells = output_reshaped[valid_mask]
        grid_indices, anchor_indices = np.where(valid_mask)

        # Apply sigmoid only to valid elements
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

        # 4. Grid generation and anchor selection
        grid_y = (grid_indices // width).astype(np.float32)
        grid_x = (grid_indices % width).astype(np.float32)
        valid_anchors = anchors[anchor_indices]

        # 5. YOLOv5 anchor decoding
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

    # Map coordinates back to original image scaling
    pad_x, pad_y = pad
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale

    valid_boxes = np.maximum(valid_boxes, 0)

    # 6. Batched NMS
    max_coord = np.max(valid_boxes) + 1.0
    offsets = valid_class_ids.astype(valid_boxes.dtype) * max_coord

    # NMSBoxes needs [x1, y1, w, h]
    c_widths = valid_boxes[:, 2] - valid_boxes[:, 0]
    c_heights = valid_boxes[:, 3] - valid_boxes[:, 1]

    x1_offset = valid_boxes[:, 0] + offsets
    y1_offset = valid_boxes[:, 1] + offsets

    boxes_xywh = np.stack([x1_offset, y1_offset, c_widths, c_heights], axis=1)

    nms_indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), valid_scores.tolist(), conf_threshold, iou_threshold)

    detections = []
    if len(nms_indices) > 0:
        nms_indices = nms_indices.flatten()
        for idx in nms_indices:
            bx1, by1, bx2, by2 = valid_boxes[idx]
            c_id = valid_class_ids[idx]
            detections.append({
                'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                'confidence': float(valid_scores[idx]),
                'class_id': int(c_id),
                'class_name': class_names.get(int(c_id), f'class_{int(c_id)}')
            })

    return detections

def get_class_color_and_text_color(class_id):
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    text_color = (255, 255, 255) if sum(bgr) < 400 else (0, 0, 0)
    return bgr, text_color

def draw_detections(img, detections, save_path=None, in_place=False):
    result_img = img if in_place else img.copy()

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']

        color, text_color = get_class_color_and_text_color(class_id)

        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prevent label from drawing outside the top image boundary
        y1_label = max(y1, label_h + 10)

        # Draw background rectangle and text
        cv2.rectangle(result_img, (x1, y1_label - label_h - 10), (x1 + label_w, y1_label), color, thickness=cv2.FILLED)
        cv2.putText(result_img, label, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness=1, lineType=cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img

def main():
    parser = argparse.ArgumentParser(description="Yolov5 Demo")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.4)
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.adla)

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
    input_h = int(tensor_attr["dims"][1])
    input_w = int(tensor_attr["dims"][2])
    input_shape = (input_h, input_w)
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess input
            input_tensor, original_img, scale, pad = preprocess(image_path, input_shape, s, zp, tensor_type)

            # Run inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess results
            detections = postprocess(outputs, input_shape, scale, pad, args.conf, args.nms)

            # Print detection results
            if detections:
                print(f"    Detected {len(detections)} objects:")
                for i, det in enumerate(detections, 1):
                    print(f"      {i}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("    No objects detected")

            # Save result image
            model_name = Path(args.adla).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")
            draw_detections(original_img, detections, str(save_path))
            print(f"    Result saved to: {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()
    print(f"=" * 60)
    print(amlnn.get_perf_info())

    # Optional visualization
    # amlnn.perf_visualize()

    # Release resources
    amlnn.uninit()

if __name__ == "__main__":
    main()
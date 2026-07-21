# -*- coding: utf-8 -*-

"""
Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import os
import glob
import argparse
import cv2
import colorsys
from pathlib import Path
from amlnn.api import AMLNN

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'doughnut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def letterbox(img, new_shape, color=(114, 114, 114)):
    shape = img.shape[:2]
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

def preprocess(img_path, new_shape, scale, zero_point, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, resize_scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0

    # Quantize using the input tensor metadata and saturate to its real range.
    if tensor_type == 0:
        input_tensor = normalized_img
    else:
        raw_value = np.round(normalized_img / scale + zero_point)
        if tensor_type == 2:
            input_tensor = np.clip(raw_value, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_value, 0, 255).astype(np.uint8)
        elif tensor_type == 4:
            input_tensor = np.clip(raw_value, -32768, 32767).astype(np.int16)
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")

    return np.expand_dims(input_tensor, axis=0), original_img, resize_scale, pad

def postprocess(outputs, input_shape, scale, pad, conf_threshold=0.25, iou_threshold=0.45):
    output = np.asarray(outputs[0])
    if output.shape != (1, 1, 8400, 85):
        raise ValueError(f"Unexpected YOLOX output shape: {output.shape}; expected (1, 1, 8400, 85)")

    predictions = output[0, 0]
    input_h, input_w = input_shape
    all_boxes = []
    all_scores = []
    all_class_ids = []
    cell_offset = 0

    # YOLOX concatenates the stride-8, stride-16 and stride-32 predictions.
    for stride in (8, 16, 32):
        grid_h = input_h // stride
        grid_w = input_w // stride
        num_cells = grid_h * grid_w
        head = predictions[cell_offset:cell_offset + num_cells]
        cell_offset += num_cells

        objectness = np.clip(head[:, 4], 0.0, 1.0)
        class_probabilities = np.clip(head[:, 5:], 0.0, 1.0)
        class_ids = np.argmax(class_probabilities, axis=1)
        class_scores = class_probabilities[np.arange(num_cells), class_ids]
        scores = objectness * class_scores
        valid_indices = np.where(scores > conf_threshold)[0]
        if len(valid_indices) == 0:
            continue

        valid_head = head[valid_indices]
        grid_x = (valid_indices % grid_w).astype(np.float32)
        grid_y = (valid_indices // grid_w).astype(np.float32)
        center_x = (valid_head[:, 0] + grid_x) * stride
        center_y = (valid_head[:, 1] + grid_y) * stride
        width = np.exp(np.clip(valid_head[:, 2], -20.0, 20.0)) * stride
        height = np.exp(np.clip(valid_head[:, 3], -20.0, 20.0)) * stride

        x1 = center_x - width * 0.5
        y1 = center_y - height * 0.5
        x2 = center_x + width * 0.5
        y2 = center_y + height * 0.5
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores[valid_indices])
        all_class_ids.append(class_ids[valid_indices])

    if cell_offset != predictions.shape[0]:
        raise ValueError(f"Input shape {input_shape} generates {cell_offset} cells, not {predictions.shape[0]}")
    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    # Undo the letterbox padding and resize scale before class-aware NMS.
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes = np.maximum(boxes, 0.0)
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    selected_indices = []
    for class_id in np.unique(class_ids):
        candidate_indices = np.where(class_ids == class_id)[0]
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_xywh[candidate_indices].tolist(), scores[candidate_indices].tolist(),
            conf_threshold, iou_threshold
        )
        if len(nms_indices) > 0:
            selected_indices.extend(candidate_indices[nms_indices.flatten()].tolist())

    selected_indices.sort(key=lambda index: float(scores[index]), reverse=True)
    detections = []
    for index in selected_indices:
        class_id = int(class_ids[index])
        detections.append({
            'bbox': boxes[index].tolist(),
            'confidence': float(scores[index]),
            'class_id': class_id,
            'class_name': COCO_CLASSES[class_id]
        })
    return detections

def get_class_color(class_id):
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
    return int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)

def draw_detections(img, detections, save_path):
    result_img = img.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection['bbox']]
        color = get_class_color(detection['class_id'])
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_x = max(0, x1)
        label_y = max(y1, label_h + 10)
        cv2.rectangle(result_img, (label_x, label_y - label_h - 10), (label_x + label_w, label_y), color, cv2.FILLED)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, result_img)
    return result_img

def main():
    parser = argparse.ArgumentParser(description='YOLOX Demo')
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--nms', type=float, default=0.45)
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode='native', enable_perf=True)
    amlnn.load_model(path=args.model_path)
    tensor_info = amlnn.get_tensor_info()
    print(amlnn.get_sdk_version())

    image_files = []
    for extension in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        amlnn.uninit()
        return 0

    tensor_attr = tensor_info['inputs'][0]
    input_h = int(tensor_attr['dims'][1])
    input_w = int(tensor_attr['dims'][2])
    input_shape = (input_h, input_w)
    input_scale = float(tensor_attr['scale'])
    input_zero_point = int(tensor_attr['zp'])
    tensor_type = int(tensor_attr['type'])

    for image_idx, image_path in enumerate(image_files, 1):
        print('=' * 60)
        print(f"Processing image {image_idx}/{len(image_files)}: {os.path.basename(image_path)}")
        print('=' * 60)

        try:
            input_tensor, original_img, resize_scale, pad = preprocess(
                image_path, input_shape, input_scale, input_zero_point, tensor_type
            )
            outputs = amlnn.inference(inputs=[input_tensor])
            detections = postprocess(
                outputs, input_shape, resize_scale, pad, args.conf, args.nms
            )

            if detections:
                print(f"    Detected {len(detections)} objects:")
                for i, det in enumerate(detections, 1):
                    print(f"      {i}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("    No objects detected")

            result_dir = f"{Path(args.model_path).stem}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(original_img, detections, save_path)
            print(f"Result saved to: {save_path}")
        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")

        print()

    print("=" * 60)
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()
    return 0

if __name__ == '__main__':
    main()
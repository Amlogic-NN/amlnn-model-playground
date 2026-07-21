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

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
REG_MAX = 16
NUM_CLASSES = 80
STRIDES = (8, 16, 32)

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'doughnut',
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess(img_path, input_shape, scale, zero_point, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    input_h, input_w = input_shape
    original_h, original_w = original_img.shape[:2]
    resized_img = cv2.resize(original_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    normalized_img = (rgb_img.astype(np.float32) - MEAN) / STD

    # Quantize the ImageNet-normalized input with the model tensor metadata.
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

    scale_x = input_w / original_w
    scale_y = input_h / original_h
    return np.expand_dims(input_tensor, axis=0), original_img, scale_x, scale_y

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

def postprocess(
    outputs, input_shape, scale_x, scale_y, conf_threshold, iou_threshold, reg_max=REG_MAX
):
    if len(outputs) != 6:
        raise ValueError(f"Expected 6 PP-YOLOE outputs, got {len(outputs)}")

    input_h, input_w = input_shape
    num_bins = reg_max + 1
    safe_threshold = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inverse_threshold = np.log(safe_threshold / (1.0 - safe_threshold))
    projection = np.arange(num_bins, dtype=np.float32).reshape(1, num_bins, 1)
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Output pairs are [DFL, classes] for strides 8, 16 and 32.
    for output_idx, stride in enumerate(STRIDES):
        grid_h = input_h // stride
        grid_w = input_w // stride
        num_cells = grid_h * grid_w
        dfl_output = np.asarray(outputs[output_idx * 2])
        class_output = np.asarray(outputs[output_idx * 2 + 1])
        expected_dfl_shape = (1, num_cells, num_bins, 4)
        expected_class_shape = (1, grid_h, grid_w, NUM_CLASSES)

        if dfl_output.shape != expected_dfl_shape:
            raise ValueError(
                f"Unexpected stride-{stride} DFL shape: {dfl_output.shape}; "
                f"expected {expected_dfl_shape}"
            )
        if class_output.shape != expected_class_shape:
            raise ValueError(
                f"Unexpected stride-{stride} class shape: {class_output.shape}; "
                f"expected {expected_class_shape}"
            )

        class_logits = class_output[0].reshape(num_cells, NUM_CLASSES)
        max_class_logits = np.max(class_logits, axis=1)
        valid_indices = np.where(max_class_logits > inverse_threshold)[0]
        if len(valid_indices) == 0:
            continue

        valid_class_logits = class_logits[valid_indices]
        scores = sigmoid(max_class_logits[valid_indices])
        class_ids = np.argmax(valid_class_logits, axis=1)

        # reg_max=16 means 17 bins representing distances 0 through 16.
        dfl_logits = dfl_output[0, valid_indices].copy()
        dfl_logits -= np.max(dfl_logits, axis=1, keepdims=True)
        probabilities = np.exp(dfl_logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        distances = np.sum(probabilities * projection, axis=1)

        grid_x = (valid_indices % grid_w).astype(np.float32)
        grid_y = (valid_indices // grid_w).astype(np.float32)
        center_x = (grid_x + 0.5) * stride
        center_y = (grid_y + 0.5) * stride
        x1 = center_x - distances[:, 0] * stride
        y1 = center_y - distances[:, 1] * stride
        x2 = center_x + distances[:, 2] * stride
        y2 = center_y + distances[:, 3] * stride

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores)
        all_class_ids.append(class_ids)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    # Reverse the direct resize independently along X and Y.
    boxes[:, [0, 2]] /= scale_x
    boxes[:, [1, 3]] /= scale_y
    boxes = np.maximum(boxes, 0.0)
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    selected_indices = []
    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_xywh[class_indices].tolist(), scores[class_indices].tolist(),
            conf_threshold, iou_threshold
        )
        if len(nms_indices) > 0:
            selected_indices.extend(class_indices[nms_indices.flatten()].tolist())

    selected_indices.sort(key=lambda index: float(scores[index]), reverse=True)
    detections = []
    for index in selected_indices:
        class_id = int(class_ids[index])
        detections.append({
            'bbox': boxes[index].tolist(),
            'confidence': float(scores[index]),
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id]
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
    parser = argparse.ArgumentParser(description='PP-YOLOE Demo')
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.6)
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
            input_tensor, original_img, scale_x, scale_y = preprocess(
                image_path, input_shape, input_scale, input_zero_point, tensor_type
            )
            outputs = amlnn.inference(inputs=[input_tensor])
            detections = postprocess(
                outputs, input_shape, scale_x, scale_y, REG_MAX,
                args.conf, args.nms
            )

            print(f"Detected {len(detections)} objects")
            result_dir = f"{Path(args.model_path).stem}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(original_img, detections, save_path)
            print(f"Result saved to: {save_path}")
        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")

    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()
    return 0

if __name__ == '__main__':
    main()
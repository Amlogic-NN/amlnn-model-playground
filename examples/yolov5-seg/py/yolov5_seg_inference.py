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

STRIDES = (8, 16, 32)
ANCHORS = np.array([
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]]
], dtype=np.float32)

NUM_MASK_COEFFICIENTS = 32

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

    # Quantize with the input tensor metadata and saturate to its real range.
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

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

def postprocess(outputs, input_shape, scale, pad, conf_threshold, iou_threshold):
    input_h, input_w = input_shape
    values_per_anchor = 5 + len(CLASS_NAMES) + NUM_MASK_COEFFICIENTS
    expected_channels = 3 * values_per_anchor
    safe_threshold = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inverse_threshold = np.log(safe_threshold / (1.0 - safe_threshold))
    all_boxes = []
    all_scores = []
    all_class_ids = []
    all_mask_coefficients = []

    # Each NHWC cell contains three anchors with 117 values per anchor.
    for output_idx, stride in enumerate(STRIDES):
        output = np.asarray(outputs[output_idx])
        grid_h = input_h // stride
        grid_w = input_w // stride
        expected_shape = (1, grid_h, grid_w, expected_channels)
        if output.shape != expected_shape:
            raise ValueError(
                f"Unexpected stride-{stride} output shape: {output.shape}; "
                f"expected {expected_shape}"
            )

        predictions = output[0].reshape(grid_h * grid_w, 3, values_per_anchor)
        objectness_logits = predictions[:, :, 4]
        candidate_cells, candidate_anchors = np.where(objectness_logits > inverse_threshold)
        if len(candidate_cells) == 0:
            continue

        candidates = predictions[candidate_cells, candidate_anchors]
        class_probabilities = sigmoid(candidates[:, 5:5 + len(CLASS_NAMES)])
        class_ids = np.argmax(class_probabilities, axis=1)
        class_scores = class_probabilities[np.arange(len(candidates)), class_ids]
        scores = sigmoid(candidates[:, 4]) * class_scores
        valid_mask = scores > conf_threshold
        if not np.any(valid_mask):
            continue

        candidates = candidates[valid_mask]
        cell_indices = candidate_cells[valid_mask]
        anchor_indices = candidate_anchors[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]

        grid_x = (cell_indices % grid_w).astype(np.float32)
        grid_y = (cell_indices // grid_w).astype(np.float32)
        box_values = sigmoid(candidates[:, :4])
        center_x = (box_values[:, 0] * 2.0 - 0.5 + grid_x) * stride
        center_y = (box_values[:, 1] * 2.0 - 0.5 + grid_y) * stride
        anchor_values = ANCHORS[output_idx, anchor_indices]
        width = np.square(box_values[:, 2] * 2.0) * anchor_values[:, 0]
        height = np.square(box_values[:, 3] * 2.0) * anchor_values[:, 1]

        x1 = center_x - width * 0.5
        y1 = center_y - height * 0.5
        x2 = center_x + width * 0.5
        y2 = center_y + height * 0.5
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores)
        all_class_ids.append(class_ids)
        all_mask_coefficients.append(candidates[:, 5 + len(CLASS_NAMES):].copy())

    prototype_mask = np.asarray(outputs[3])
    if prototype_mask.shape != (1, 160, 160, NUM_MASK_COEFFICIENTS):
        raise ValueError(f"Unexpected prototype output shape: {prototype_mask.shape}")
    prototype_mask = prototype_mask[0]

    if not all_boxes:
        return [], prototype_mask

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)
    mask_coefficients = np.concatenate(all_mask_coefficients, axis=0)

    # Undo letterbox padding and scaling before class-aware NMS.
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
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
            'class_name': CLASS_NAMES[class_id],
            'mask_coefficients': mask_coefficients[index].copy()
        })
    return detections, prototype_mask

def draw_mask(img, mask_coefficients, prototype_mask, bbox, input_shape, scale, pad, color, alpha):
    original_h, original_w = img.shape[:2]
    input_h, input_w = input_shape
    pad_x, pad_y = pad

    # Contract the 32 coefficients against the NHWC prototype channels.
    mask = np.tensordot(prototype_mask, mask_coefficients, axes=([2], [0]))
    mask = sigmoid(mask)
    mask_input = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    crop_x1 = int(pad_x)
    crop_y1 = int(pad_y)
    crop_x2 = min(input_w, crop_x1 + resized_w)
    crop_y2 = min(input_h, crop_y1 + resized_h)
    mask_cropped = mask_input[crop_y1:crop_y2, crop_x1:crop_x2]
    if mask_cropped.size == 0:
        return img

    final_mask = cv2.resize(mask_cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = final_mask > 0.5

    # Prevent each prototype reconstruction from spilling outside its retained box.
    box_x1 = max(0, int(np.floor(bbox[0])))
    box_y1 = max(0, int(np.floor(bbox[1])))
    box_x2 = min(original_w, int(np.ceil(bbox[2])))
    box_y2 = min(original_h, int(np.ceil(bbox[3])))
    bbox_mask = np.zeros((original_h, original_w), dtype=bool)
    if box_x2 > box_x1 and box_y2 > box_y1:
        bbox_mask[box_y1:box_y2, box_x1:box_x2] = True
    binary_mask &= bbox_mask

    if not np.any(binary_mask):
        return img
    colored_mask = np.zeros_like(img, dtype=np.uint8)
    colored_mask[binary_mask] = color
    blended = cv2.addWeighted(img, 1.0 - alpha, colored_mask, alpha, 0)
    result = img.copy()
    result[binary_mask] = blended[binary_mask]
    return result

def get_class_color_and_text_color(class_id):
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    text_color = (255, 255, 255) if sum(bgr) < 400 else (0, 0, 0)
    return bgr, text_color

def draw_detections(img, detections, prototype_mask, input_shape, scale, pad, mask_alpha, save_path=None):
    result_img = img.copy()

    # Draw masks first so boxes and labels remain visible.
    for detection in detections:
        color, _ = get_class_color_and_text_color(detection['class_id'])
        result_img = draw_mask(
            result_img, detection['mask_coefficients'], prototype_mask,
            detection['bbox'], input_shape, scale, pad, color, mask_alpha
        )

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection['bbox']]
        color, text_color = get_class_color_and_text_color(detection['class_id'])
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_x = max(0, x1)
        label_y = max(y1, label_h + 10)
        cv2.rectangle(result_img, (label_x, label_y - label_h - 10), (label_x + label_w, label_y), color, cv2.FILLED)
        cv2.putText(result_img, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, result_img)
    return result_img

def main():
    parser = argparse.ArgumentParser(description='YOLOv5-Seg Demo')
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--mask-alpha', type=float, default=0.5)
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
            detections, prototype_mask = postprocess(
                outputs, input_shape, resize_scale, pad, args.conf, args.nms
            )

            if detections:
                print(f"    Detected {len(detections)} objects:")
                for detection_idx, detection in enumerate(detections, 1):
                    print(
                        f"      {detection_idx}. {detection['class_name']} "
                        f"({detection['confidence']:.2f})"
                    )
            else:
                print("    No objects detected")

            result_dir = f"{Path(args.model_path).stem}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(
                original_img, detections, prototype_mask, input_shape,
                resize_scale, pad, args.mask_alpha, save_path
            )
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
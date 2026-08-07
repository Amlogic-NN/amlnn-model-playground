"""
Copyright (C) 2026 Amlogic, Inc. All rights reserved.

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

class_names = {
    0: 'plane',
    1: 'ship',
    2: 'storage tank',
    3: 'baseball diamond',
    4: 'tennis court',
    5: 'basketball court',
    6: 'ground track field',
    7: 'harbor',
    8: 'bridge',
    9: 'large vehicle',
    10: 'small vehicle',
    11: 'helicopter',
    12: 'roundabout',
    13: 'soccer ball field',
    14: 'swimming pool'
}

REG_MAX = 16

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

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
    rgb_float = rgb_img.astype(np.float32)

    # Normalize float input or quantize with the model's scale and zero point.
    if tensor_type == 0:
        input_tensor = rgb_float / 255.0
    elif tensor_type in (2, 3, 4):
        inv_scale = np.float32(1.0 / (255.0 * scale))
        raw_value = np.round(rgb_float * inv_scale + zero_point)

        if tensor_type == 2:
            input_tensor = np.clip(raw_value, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_value, 0, 255).astype(np.uint8)
        else:
            input_tensor = np.clip(raw_value, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, original_img, resize_scale, pad

def postprocess(outputs, scale, pad, conf_threshold, iou_threshold, reg_max=REG_MAX):
    if len(outputs) != 9:
        raise ValueError(f"Expected 9 YOLOv8-OBB outputs, got {len(outputs)}")

    num_classes = len(class_names)
    safe_threshold = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inverse_threshold = np.log(safe_threshold / (1.0 - safe_threshold))
    projection = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Output order: [DFL_8, ANGLE_8, CLS_8, DFL_16, ANGLE_16, CLS_16, DFL_32, ANGLE_32, CLS_32].
    strides = [8, 16, 32]
    for output_idx, stride in enumerate(strides):
        dfl_output = np.squeeze(outputs[output_idx * 3])
        angle_output = np.squeeze(outputs[output_idx * 3 + 1])
        class_output = np.squeeze(outputs[output_idx * 3 + 2])

        if dfl_output.ndim != 3:
            raise ValueError(f"Unexpected DFL shape for stride {stride}: {outputs[output_idx * 3].shape}")

        grid_h, grid_w, dfl_channels = dfl_output.shape
        num_cells = grid_h * grid_w

        if dfl_channels != 4 * reg_max:
            raise ValueError(f"Unexpected DFL shape for stride {stride}: {outputs[output_idx * 3].shape}")
        if angle_output.shape != (grid_h, grid_w):
            raise ValueError(f"Unexpected angle shape for stride {stride}: {outputs[output_idx * 3 + 1].shape}")
        if class_output.shape != (grid_h, grid_w, num_classes):
            raise ValueError(f"Unexpected class shape for stride {stride}: {outputs[output_idx * 3 + 2].shape}")

        dfl_output = dfl_output.reshape(num_cells, 4 * reg_max).T
        angle_output = angle_output.reshape(num_cells)
        class_output = class_output.reshape(num_cells, num_classes).T

        max_class_logits = np.max(class_output, axis=0)
        class_ids = np.argmax(class_output, axis=0)
        valid_indices = np.where(max_class_logits > inverse_threshold)[0]
        if len(valid_indices) == 0:
            continue

        scores = sigmoid(max_class_logits[valid_indices])
        valid_class_ids = class_ids[valid_indices]

        # Decode DFL into left, top, right, and bottom distances.
        valid_dfl = dfl_output[:, valid_indices].reshape(4, reg_max, -1)
        valid_dfl -= np.max(valid_dfl, axis=1, keepdims=True)
        probabilities = np.exp(valid_dfl)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        distances = np.sum(probabilities * projection, axis=1)

        left, top, right, bottom = distances
        grid_x = (valid_indices % grid_w).astype(np.float32)
        grid_y = (valid_indices // grid_w).astype(np.float32)

        # YOLOv8-OBB angle range: [-pi/4, 3pi/4].
        angles = (sigmoid(angle_output[valid_indices]) - 0.25) * np.pi
        cos_angle = np.cos(angles)
        sin_angle = np.sin(angles)

        # Rotate the offset from the grid anchor to the box center.
        offset_x = (right - left) * 0.5
        offset_y = (bottom - top) * 0.5
        center_x = (grid_x + 0.5 + offset_x * cos_angle - offset_y * sin_angle) * stride
        center_y = (grid_y + 0.5 + offset_x * sin_angle + offset_y * cos_angle) * stride
        width = (left + right) * stride
        height = (top + bottom) * stride

        all_boxes.append(np.stack([center_x, center_y, width, height, angles], axis=1))
        all_scores.append(scores)
        all_class_ids.append(valid_class_ids)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    # Undo letterbox padding and scaling. Angles remain unchanged.
    pad_x, pad_y = pad
    boxes[:, 0] = (boxes[:, 0] - pad_x) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_y) / scale
    boxes[:, 2] /= scale
    boxes[:, 3] /= scale

    valid_size_mask = (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
    boxes = boxes[valid_size_mask]
    scores = scores[valid_size_mask]
    class_ids = class_ids[valid_size_mask]

    if len(boxes) == 0:
        return []

    # Run rotated NMS separately for each class.
    selected_indices = []
    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        rotated_rectangles = [
            (
                (float(boxes[idx, 0]), float(boxes[idx, 1])),
                (float(boxes[idx, 2]), float(boxes[idx, 3])),
                float(boxes[idx, 4] * 180.0 / np.pi)
            )
            for idx in class_indices
        ]

        nms_indices = cv2.dnn.NMSBoxesRotated(
            rotated_rectangles,
            scores[class_indices].tolist(),
            conf_threshold,
            iou_threshold
        )

        if len(nms_indices) > 0:
            selected_indices.extend(class_indices[np.asarray(nms_indices).reshape(-1)].tolist())

    selected_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)

    detections = []
    for detection_idx in selected_indices:
        center_x, center_y, width, height, angle_rad = boxes[detection_idx]
        angle_degrees = float(angle_rad * 180.0 / np.pi)
        corners = cv2.boxPoints((
            (float(center_x), float(center_y)),
            (float(width), float(height)),
            angle_degrees
        ))

        class_id = int(class_ids[detection_idx])
        detections.append({
            "bbox": corners.tolist(),
            "center": [float(center_x), float(center_y)],
            "size": [float(width), float(height)],
            "angle": float(angle_rad),
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            "score": float(scores[detection_idx])
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

    for detection in detections:
        corners = np.asarray(detection['bbox'], dtype=np.int32)
        class_id = detection['class_id']
        class_name = detection['class_name']
        score = detection['score']
        color, text_color = get_class_color_and_text_color(class_id)

        cv2.polylines(result_img, [corners], isClosed=True, color=color, thickness=2)
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        top_corner = corners[np.argmin(corners[:, 1])]
        label_x = max(0, int(top_corner[0]))
        label_y = max(int(top_corner[1]), label_h + 10)
        cv2.rectangle(
            result_img, (label_x, label_y - label_h - 10),
            (label_x + label_w, label_y), color, cv2.FILLED
        )
        cv2.putText(
            result_img, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, text_color, thickness=1, lineType=cv2.LINE_AA
        )

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img

def main():
    parser = argparse.ArgumentParser(description="YOLOv8-OBB Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.45)
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)
    tensor_info = amlnn.get_tensor_info()
    print(amlnn.get_sdk_version())

    image_files = []
    for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        amlnn.uninit()
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")
    for image_path in image_files:
        print(f"  - {os.path.basename(image_path)}")
    print()

    tensor_attr = tensor_info["inputs"][0]
    input_h = int(tensor_attr["dims"][1])
    input_w = int(tensor_attr["dims"][2])
    input_shape = (input_h, input_w)
    input_scale = float(tensor_attr["scale"])
    input_zero_point = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    for image_idx, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {image_idx}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            input_tensor, original_img, resize_scale, pad = preprocess(
                image_path, input_shape, input_scale, input_zero_point, tensor_type
            )
            outputs = amlnn.inference(inputs=[input_tensor])
            detections = postprocess(outputs, resize_scale, pad, args.conf, args.nms)

            if detections:
                print(f"    Detected {len(detections)} objects:")
                for detection_idx, detection in enumerate(detections, 1):
                    print(
                        f"      {detection_idx}. {detection['class_name']} "
                        f"({detection['score']:.2f})"
                    )
            else:
                print("    No objects detected")

            result_dir = f"{Path(args.model_path).stem}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(original_img, detections, save_path)
            print(f"    Result saved to: {save_path}")
        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")

        print()

    print("=" * 60)
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
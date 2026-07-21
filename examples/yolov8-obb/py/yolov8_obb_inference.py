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

def postprocess(outputs, scale, pad, conf_threshold, iou_threshold):
    if len(outputs) != 3:
        raise ValueError(f"Expected 3 YOLOv8-OBB outputs, got {len(outputs)}")

    # Fixed output layout: [1, 1, 4, N], [1, 1, 15, N], and [1, 1, 1, N].
    bboxes = np.squeeze(outputs[0]).T
    class_scores = np.squeeze(outputs[1]).T
    angles = np.squeeze(outputs[2])

    num_predictions = bboxes.shape[0]
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(f"Unexpected bbox output shape: {outputs[0].shape}")
    if class_scores.shape != (num_predictions, len(class_names)):
        raise ValueError(f"Unexpected class output shape: {outputs[1].shape}")
    if angles.ndim != 1 or angles.shape[0] != num_predictions:
        raise ValueError(f"Unexpected angle output shape: {outputs[2].shape}")

    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    valid_indices = np.where(scores > conf_threshold)[0]
    if len(valid_indices) == 0:
        return []

    bboxes = bboxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]
    angles = angles[valid_indices]
    pad_x, pad_y = pad

    corners_list = []
    boxes_xywh = []
    for bbox, angle_rad in zip(bboxes, angles):
        center_x = (float(bbox[0]) - pad_x) / scale
        center_y = (float(bbox[1]) - pad_y) / scale
        width = float(bbox[2]) / scale
        height = float(bbox[3]) / scale
        angle_degrees = float(angle_rad) * 180.0 / np.pi

        corners = cv2.boxPoints(((center_x, center_y), (width, height), angle_degrees))
        corners_list.append(corners)

        # Use the enclosing axis-aligned rectangle for OpenCV NMS.
        x_min = float(np.min(corners[:, 0]))
        y_min = float(np.min(corners[:, 1]))
        x_max = float(np.max(corners[:, 0]))
        y_max = float(np.max(corners[:, 1]))
        boxes_xywh.append([x_min, y_min, x_max - x_min, y_max - y_min])

    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
    selected_indices = []

    # Run NMS separately for each class.
    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_xywh[class_indices].tolist(),
            scores[class_indices].tolist(),
            conf_threshold,
            iou_threshold
        )

        if len(nms_indices) > 0:
            selected_indices.extend(class_indices[nms_indices.flatten()].tolist())

    selected_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)

    detections = []
    for detection_idx in selected_indices:
        class_id = int(class_ids[detection_idx])
        detections.append({
            'bbox': corners_list[detection_idx].tolist(),
            'class_id': class_id,
            'class_name': class_names.get(class_id, f'class_{class_id}'),
            'score': float(scores[detection_idx])
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
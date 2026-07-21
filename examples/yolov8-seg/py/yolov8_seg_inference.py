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
    if len(outputs) != 4:
        raise ValueError(f"Expected 4 YOLOv8-Seg outputs, got {len(outputs)}")

    # Fixed decoded layouts: [4,N], [80,N], [32,N], and [160,160,32].
    bboxes = np.squeeze(outputs[0]).T
    class_scores = np.squeeze(outputs[1]).T
    mask_coefficients = np.squeeze(outputs[2]).T
    prototype_mask = np.squeeze(outputs[3])

    num_predictions = bboxes.shape[0]
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(f"Unexpected bbox output shape: {outputs[0].shape}")
    if class_scores.shape != (num_predictions, len(CLASS_NAMES)):
        raise ValueError(f"Unexpected class output shape: {outputs[1].shape}")
    if mask_coefficients.shape != (num_predictions, 32):
        raise ValueError(f"Unexpected mask coefficient shape: {outputs[2].shape}")
    if prototype_mask.shape != (160, 160, 32):
        raise ValueError(f"Unexpected prototype output shape: {outputs[3].shape}")

    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    valid_indices = np.where(scores > conf_threshold)[0]
    if len(valid_indices) == 0:
        return [], prototype_mask

    bboxes = bboxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]
    mask_coefficients = mask_coefficients[valid_indices]

    # Convert decoded XYWH boxes back to original-image coordinates.
    pad_x, pad_y = pad
    center_x = bboxes[:, 0]
    center_y = bboxes[:, 1]
    width = bboxes[:, 2]
    height = bboxes[:, 3]
    x1 = (center_x - width / 2.0 - pad_x) / scale
    y1 = (center_y - height / 2.0 - pad_y) / scale
    x2 = (center_x + width / 2.0 - pad_x) / scale
    y2 = (center_y + height / 2.0 - pad_y) / scale
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xywh = np.stack([x1, y1, width / scale, height / scale], axis=1)

    # Run NMS separately for each class.
    selected_indices = []
    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_xywh[class_indices].tolist(), scores[class_indices].tolist(),
            conf_threshold, iou_threshold
        )
        if len(nms_indices) > 0:
            selected_indices.extend(class_indices[nms_indices.flatten()].tolist())

    selected_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)
    detections = []
    for detection_idx in selected_indices:
        class_id = int(class_ids[detection_idx])
        detections.append({
            'bbox': boxes_xyxy[detection_idx].tolist(),
            'confidence': float(scores[detection_idx]),
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'mask_coefficients': mask_coefficients[detection_idx].copy()
        })

    return detections, prototype_mask

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

def draw_mask(
    img, mask_coefficients, prototype_mask, bbox, input_shape,
    scale, pad, color, alpha
):
    original_h, original_w = img.shape[:2]
    input_h, input_w = input_shape
    pad_x, pad_y = pad

    # Prototype is NHWC, so coefficients contract against its final axis.
    mask = np.tensordot(prototype_mask, mask_coefficients, axes=([2], [0]))
    mask = sigmoid(mask)
    mask_input = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    # Remove the exact letterbox region, including asymmetric one-pixel padding.
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

    # YOLOv8 masks are cropped to their corresponding detection boxes.
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

def draw_detections(
    img, detections, prototype_mask, input_shape, scale, pad,
    mask_alpha=0.5, save_path=None, in_place=False
):
    result_img = img if in_place else img.copy()

    # Draw masks first so bounding boxes and labels remain visible.
    for detection in detections:
        color, _ = get_class_color_and_text_color(detection['class_id'])
        result_img = draw_mask(
            result_img, detection['mask_coefficients'], prototype_mask,
            detection['bbox'], input_shape, scale, pad, color, mask_alpha
        )

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection['bbox']]
        confidence = detection['confidence']
        class_name = detection['class_name']
        color, text_color = get_class_color_and_text_color(detection['class_id'])

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_x = max(0, x1)
        label_y = max(y1, label_h + 10)
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
    parser = argparse.ArgumentParser(description="YOLOv8-Seg Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--mask-alpha', type=float, default=0.5)
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
            detections, prototype_mask = postprocess(
                outputs, resize_scale, pad, args.conf, args.nms
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
                resize_scale, pad, args.mask_alpha, save_path=save_path
            )
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
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
from pathlib import Path
from amlnn.api import AMLNN

KEYPOINT_NAMES = [
    'nose', 'left eye', 'right eye', 'left ear', 'right ear',
    'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
    'left wrist', 'right wrist', 'left hip', 'right hip',
    'left knee', 'right knee', 'left ankle', 'right ankle'
]

# Standard COCO-17 pose connections using zero-based keypoint indices.
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
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
        raise ValueError(f"Expected 4 YOLOv8-Pose outputs, got {len(outputs)}")

    # Fixed decoded layouts: [4,N], [N], [N,17], and [2,N,17].
    bboxes = np.squeeze(outputs[0]).T
    confidences = np.squeeze(outputs[1])
    keypoint_confidences = np.squeeze(outputs[2])
    keypoints = np.squeeze(outputs[3]).transpose(1, 2, 0)

    num_predictions = bboxes.shape[0]
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(f"Unexpected bbox output shape: {outputs[0].shape}")
    if confidences.ndim != 1 or confidences.shape[0] != num_predictions:
        raise ValueError(f"Unexpected confidence output shape: {outputs[1].shape}")
    if keypoint_confidences.shape != (num_predictions, len(KEYPOINT_NAMES)):
        raise ValueError(f"Unexpected keypoint confidence shape: {outputs[2].shape}")
    if keypoints.shape != (num_predictions, len(KEYPOINT_NAMES), 2):
        raise ValueError(f"Unexpected keypoint coordinate shape: {outputs[3].shape}")

    valid_indices = np.where(confidences > conf_threshold)[0]
    if len(valid_indices) == 0:
        return []

    bboxes = bboxes[valid_indices]
    confidences = confidences[valid_indices]
    keypoints = keypoints[valid_indices].copy()
    keypoint_confidences = keypoint_confidences[valid_indices]

    # Convert decoded XYWH boxes and keypoints back to original-image coordinates.
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

    keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / scale
    keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / scale

    nms_indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), confidences.tolist(), conf_threshold, iou_threshold
    )

    detections = []
    if len(nms_indices) > 0:
        for detection_idx in nms_indices.flatten():
            detections.append({
                'bbox': boxes_xyxy[detection_idx].tolist(),
                'confidence': float(confidences[detection_idx]),
                'keypoints': keypoints[detection_idx].tolist(),
                'keypoint_confidences': keypoint_confidences[detection_idx].tolist()
            })

    return detections

def draw_pose(img, keypoints, keypoint_confidences, keypoint_threshold):
    img_height, img_width = img.shape[:2]

    # Draw skeleton lines before points so keypoints remain visible.
    for start_idx, end_idx in SKELETON:
        if (keypoint_confidences[start_idx] <= keypoint_threshold or
            keypoint_confidences[end_idx] <= keypoint_threshold):
            continue

        x1, y1 = keypoints[start_idx]
        x2, y2 = keypoints[end_idx]
        if not (0 <= x1 < img_width and 0 <= y1 < img_height and
                0 <= x2 < img_width and 0 <= y2 < img_height):
            continue

        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    for keypoint_idx, (x, y) in enumerate(keypoints):
        if keypoint_confidences[keypoint_idx] <= keypoint_threshold:
            continue
        if not (0 <= x < img_width and 0 <= y < img_height):
            continue

        point = (int(x), int(y))
        cv2.circle(img, point, 4, (0, 0, 255), cv2.FILLED)
        cv2.putText(
            img, KEYPOINT_NAMES[keypoint_idx], (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )

def draw_detections(
    img, detections, keypoint_threshold=0.5, save_path=None, in_place=False
):
    result_img = img if in_place else img.copy()

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection['bbox']]
        confidence = detection['confidence']
        keypoints = detection['keypoints']
        keypoint_confidences = detection['keypoint_confidences']
        color = (230, 46, 46)

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"person: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_x = max(0, x1)
        label_y = max(y1, label_h + 10)
        cv2.rectangle(
            result_img, (label_x, label_y - label_h - 10),
            (label_x + label_w, label_y), color, cv2.FILLED
        )
        cv2.putText(
            result_img, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        draw_pose(result_img, keypoints, keypoint_confidences, keypoint_threshold)

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img

def main():
    parser = argparse.ArgumentParser(description="YOLOv8-Pose Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--keypoint-conf', type=float, default=0.5)
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
                print(f"    Detected {len(detections)} people:")
                for detection_idx, detection in enumerate(detections, 1):
                    visible_keypoints = sum(
                        score > args.keypoint_conf
                        for score in detection['keypoint_confidences']
                    )
                    print(
                        f"      {detection_idx}. confidence={detection['confidence']:.2f}, "
                        f"visible keypoints={visible_keypoints}/17"
                    )
            else:
                print("    No people detected")

            result_dir = f"{Path(args.model_path).stem}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(
                original_img, detections, args.keypoint_conf, save_path=save_path
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
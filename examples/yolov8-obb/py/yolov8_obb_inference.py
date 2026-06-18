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

MEAN = np.array([0, 0, 0], dtype=np.float32)
STD  = np.array([255, 255, 255], dtype=np.float32)

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


def letterbox(img, new_shape=(1024, 1024), color=(114, 114, 114)):
    shape = img.shape[:2]  # [height, width]
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

def preprocess(img_path, new_shape=(1024, 1024), data_format='NHWC', s=0.003789, zp=-128, tensor_type=2):
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


def postprocess(outputs, scale, pad, data_format='NCHW', strides=[8, 16, 32], conf_threshold=0.6, iou_threshold=0.4):
    # 1. Extract and shape properly
    bboxes = np.squeeze(outputs[0]).T           # (21504, 4)
    class_scores = np.squeeze(outputs[1]).T     # (21504, 15)
    angle = np.squeeze(outputs[2])              # (21504)

    # 2. Filter out low confidence detections
    max_scores = np.max(class_scores, axis=1)
    mask = max_scores > conf_threshold

    filtered_bboxes = bboxes[mask]
    filtered_class_scores = class_scores[mask]
    filtered_angles = angle[mask]

    if len(filtered_bboxes) == 0:
        return []

    obb_corners = []
    boxes_xywh_scaled = []
    pad_x, pad_y = pad

    # 3. Process each detection individually
    for i in range(len(filtered_bboxes)):
        cx = float(filtered_bboxes[i, 0])
        cy = float(filtered_bboxes[i, 1])
        w = float(filtered_bboxes[i, 2])
        h = float(filtered_bboxes[i, 3])
        angle_rad = float(filtered_angles[i])

        # Undo letterbox scaling
        cx = (cx - pad_x) / scale
        cy = (cy - pad_y) / scale
        w /= scale
        h /= scale

        # Convert Radians to Degrees for OpenCV
        angle_deg = angle_rad * (180.0 / np.pi)

        # Create Rotated Rectangle and get 4 corners
        rect = ((cx, cy), (w, h), angle_deg)
        corners = cv2.boxPoints(rect)  # Returns shape (4, 2)
        obb_corners.append(corners)

        # Create axis-aligned enclosing rectangle for standard NMS
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])

        boxes_xywh_scaled.append([
            float(x_min),
            float(y_min),
            float(x_max - x_min),
            float(y_max - y_min)
        ])

    # 4. NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh_scaled,
        np.max(filtered_class_scores, axis=1).tolist(),
        conf_threshold,
        iou_threshold
    )

    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detections.append({
                'bbox': obb_corners[idx].tolist(),
                'class_id': int(np.argmax(filtered_class_scores[idx])),
                'score': float(np.max(filtered_class_scores[idx]))
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

    print(f"    Drawing {len(detections)} detections")

    for i, det in enumerate(detections):
        # Convert corners back to integers for drawing
        corners = np.array(det['bbox'], dtype=np.int32)
        class_id = det['class_id']
        score = det['score']

        print(f"    Detection {i+1}:")
        print(f"      Class: {class_names.get(class_id, 'Unknown')} | Score: {score:.4f}")

        color = get_class_color(class_id)

        # Draw Oriented Bounding Box
        cv2.polylines(
            result_img,
            [corners],
            isClosed=True,
            color=color,
            thickness=2
        )

        # Draw Label
        label = f"{class_names.get(class_id, 'Unknown')}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Find the topmost corner point to place the label nicely
        top_idx = np.argmin(corners[:, 1])
        x1, y1 = corners[top_idx]

        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(save_path, result_img)
    print(f"    Image saved to: {save_path}")
    return result_img


def main():
    parser = argparse.ArgumentParser(description="Yolov8-obb Demo")
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

    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            input_tensor, original_img, scale, pad = preprocess(
                image_path, new_shape=(1024, 1024), data_format='NHWC', s=s, zp=zp, tensor_type=tensor_type
            )

            outputs = amlnn.inference(inputs=[input_tensor])

            detections = postprocess(outputs, scale, pad, conf_threshold=0.25, iou_threshold=0.45)

            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            draw_detections(original_img, detections, str(save_path))
            print(f"    Result saved to: {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
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

import argparse
import glob
import os
from pathlib import Path
import cv2
import numpy as np
from amlnn.api import AMLNN

INPUT_SIZE = 224
NUM_COORDS = 12
MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
STD  = np.array([127.5, 127.5, 127.5], dtype=np.float32)

def letterbox(img, new_shape, color=(0, 0, 0)):
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

    # 3. BlazePose detector normalization and quantization
    normalized_img = (rgb_float - MEAN) / STD
    if tensor_type == 0:
        input_tensor = normalized_img
    elif tensor_type in (2, 3, 4):
        raw_val = np.round(normalized_img / s + zp)

        if tensor_type == 2:
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, original_img, scale, pad

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100.0, 100.0)))


def iou(a, b):
    x1, y1 = max(a[1], b[1]), max(a[0], b[0])
    x2, y2 = min(a[3], b[3]), min(a[2], b[2])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[3] - a[1]) * max(0.0, a[2] - a[0])
    area_b = max(0.0, b[3] - b[1]) * max(0.0, b[2] - b[0])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def weighted_nms(detections, iou_threshold):
    detections = sorted(detections, key=lambda det: det[-1], reverse=True)
    results = []

    while detections:
        reference = detections[0]
        group = [reference]
        remaining = []
        for det in detections[1:]:
            if iou(reference, det) > iou_threshold:
                group.append(det)
            else:
                remaining.append(det)
        detections = remaining
        weights = np.asarray([det[-1] for det in group], dtype=np.float32)
        merged = np.average(np.asarray([det[:-1] for det in group]), axis=0, weights=weights)
        results.append(np.concatenate([merged, [max(weights)]]).astype(np.float32))

    return results


def postprocess(outputs, anchors, input_shape, scale, pad, original_shape,
                conf_threshold=0.5, iou_threshold=0.3):
    input_h, input_w = input_shape
    num_anchors = anchors.shape[0]

    # Output 0: [1, 1, 2254, 12]
    # Output 1: [1, 1, 2254, 1]
    raw_boxes = np.asarray(outputs[0], dtype=np.float32).reshape(num_anchors, NUM_COORDS)
    raw_scores = np.asarray(outputs[1], dtype=np.float32).reshape(num_anchors)

    # Anchor format: [x_center, y_center, width, height]
    anc_x, anc_y, anc_w, anc_h = anchors.T

    scores = sigmoid(raw_scores)
    valid_indices = np.where(scores > conf_threshold)[0]

    detections = []

    for i in valid_indices:
        # Decode center and size using the corresponding anchor.
        x_center = raw_boxes[i, 0] / input_w * anc_w[i] + anc_x[i]
        y_center = raw_boxes[i, 1] / input_h * anc_h[i] + anc_y[i]
        width = raw_boxes[i, 2] / input_w * anc_w[i]
        height = raw_boxes[i, 3] / input_h * anc_h[i]

        coords = [
            y_center - height / 2,
            x_center - width / 2,
            y_center + height / 2,
            x_center + width / 2
        ]

        # Decode the four BlazePose detector keypoints.
        for k in range(4):
            keypoint_x = (
                raw_boxes[i, 4 + k * 2] / input_w * anc_w[i]
                + anc_x[i]
            )
            keypoint_y = (
                raw_boxes[i, 5 + k * 2] / input_h * anc_h[i]
                + anc_y[i]
            )

            coords.extend([keypoint_x, keypoint_y])

        detections.append(
            np.asarray(coords + [scores[i]], dtype=np.float32)
        )

    detections = weighted_nms(detections, iou_threshold)

    # Remove letterbox and normalize coordinates to the original image.
    orig_h, orig_w = original_shape[:2]
    pad_x, pad_y = pad

    for det in detections:
        # x coordinates: xmin, xmax and four keypoint x values
        det[[1, 3, 4, 6, 8, 10]] = (
            det[[1, 3, 4, 6, 8, 10]] * input_w - pad_x
        ) / scale / orig_w

        # y coordinates: ymin, ymax and four keypoint y values
        det[[0, 2, 5, 7, 9, 11]] = (
            det[[0, 2, 5, 7, 9, 11]] * input_h - pad_y
        ) / scale / orig_h

        # Clip the bounding box. Keypoints are intentionally not clipped
        # because the landmark ROI may extend beyond the image.
        det[:4] = np.clip(det[:4], 0.0, 1.0)

    return detections


def draw_detections(img, detections, save_path):
    result_img = img.copy()
    height, width = img.shape[:2]

    for det in detections:
        y1, x1, y2, x2 = det[:4]
        cv2.rectangle(result_img, (int(x1 * width), int(y1 * height)),
                      (int(x2 * width), int(y2 * height)), (0, 255, 0), 2)
        for k in range(4):
            x, y = det[4 + k * 2], det[5 + k * 2]
            cv2.circle(result_img, (int(x * width), int(y * height)), 4, (0, 0, 255), -1)
        cv2.putText(result_img, f"pose: {det[-1]:.2f}", (int(x1 * width), max(15, int(y1 * height) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(save_path, result_img)

def main():
    parser = argparse.ArgumentParser(description="BlazePose detector ADLA demo")
    parser.add_argument('--model-path', required=True, help='Path to detector .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--anchor-path', default="anchors.npy", help='Path to anchor.npy')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--nms', type=float, default=0.3)
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)
    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    input_attr = tensor_info["inputs"][0]
    input_h, input_w = int(input_attr["dims"][1]), int(input_attr["dims"][2])
    input_shape = (input_h, input_w)
    s = float(input_attr["scale"])
    zp = int(input_attr["zp"])
    tensor_type = int(input_attr["type"])

    if (input_h, input_w) != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError(f"This anchor configuration requires a 224x224 input, got {input_h}x{input_w}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return 0

    anchor_path = Path(args.anchor_path)
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")

    anchors = np.load(anchor_path, allow_pickle=False).astype(np.float32)

    # Accept either (2254, 4) or its transposed form (4, 2254).
    if anchors.shape == (4, 2254):
        anchors = anchors.T
    elif anchors.shape != (2254, 4):
        raise ValueError(
            f"Unexpected anchor shape: {anchors.shape}. "
            "Expected (2254, 4) or (4, 2254)."
        )

    if not np.all(np.isfinite(anchors)):
        raise ValueError("Anchor file contains NaN or infinite values.")

    print(f"Loaded {len(anchors)} anchors from: {anchor_path}")

    print(f"Found {len(image_files)} image file(s) to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    result_dir = f"{Path(args.model_path).stem}_result"
    os.makedirs(result_dir, exist_ok=True)

    for i, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)
        try:
            # Preprocess input
            input_tensor, original_img, scale, pad = preprocess(image_path, input_shape, s, zp, tensor_type)

            # Run inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess results
            detections = postprocess(outputs, anchors, input_shape, scale, pad, original_img.shape, args.conf, args.nms)

            stem = Path(image_path).stem
            txt_path = os.path.join(result_dir, f"{stem}_det.txt")

            with open(txt_path, "w", encoding="utf-8") as file:
                for det in detections:
                    file.write(" ".join(f"{float(value):.8f}" for value in det) + "\n")

            print(f"Detection data saved to: {txt_path}")

            save_path = os.path.join(result_dir, f"{stem}_result.jpg")
            draw_detections(original_img, detections, save_path)

            print(f"Result image saved to: {save_path}")
            print(f"Detected {len(detections)} pose(s) in {os.path.basename(image_path)}")

        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")
        print()

    print("=" * 60)
    print(amlnn.get_perf_info())
    amlnn.uninit()


if __name__ == "__main__":
    main()

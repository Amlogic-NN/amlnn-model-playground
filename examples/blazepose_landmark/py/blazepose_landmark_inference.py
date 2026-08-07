#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import glob
import math
import os
from pathlib import Path
import cv2
import numpy as np
from amlnn.api import AMLNN

HEATMAP_SEARCH_RADIUS = 9
HEATMAP_MIN_CONFIDENCE = 0.5
NUM_LANDMARKS = 39
NUM_POSE_LANDMARKS = 33
INPUT_SIZE = 256

SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]


def load_detections(txt_path):
    detections = []

    if not os.path.isfile(txt_path):
        return detections

    with open(txt_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            values = [float(value) for value in line.split()]

            if len(values) != 13:
                raise ValueError(f"{txt_path}:{line_number} must contain 13 values, got {len(values)}")

            detections.append(np.asarray(values, dtype=np.float32))

    return detections


def detection_to_roi(detection, image_shape):
    height, width = image_shape[:2]

    center_x = detection[4] * width
    center_y = detection[5] * height
    end_x = detection[6] * width
    end_y = detection[7] * height

    radius = math.hypot(end_x - center_x, end_y - center_y)

    if radius < 1.0:
        box_w = (detection[3] - detection[1]) * width
        box_h = (detection[2] - detection[0]) * height
        center_x = (detection[1] + detection[3]) * width / 2
        center_y = (detection[0] + detection[2]) * height / 2
        radius = max(box_w, box_h) / 2

    rotation = math.pi / 2 - math.atan2(-(end_y - center_y), end_x - center_x)

    return {
        "cx": center_x,
        "cy": center_y,
        "size": 2.5 * radius,
        "rotation": rotation
    }


def roi_to_image_point(x, y, roi):
    local_x = (x / INPUT_SIZE - 0.5) * roi["size"]
    local_y = (y / INPUT_SIZE - 0.5) * roi["size"]

    cosine = math.cos(roi["rotation"])
    sine = math.sin(roi["rotation"])

    return (
        roi["cx"] + cosine * local_x - sine * local_y,
        roi["cy"] + sine * local_x + cosine * local_y
    )


def preprocess(image, roi, new_shape, s, zp, tensor_type):
    input_h, input_w = new_shape

    source = np.asarray([
        roi_to_image_point(0.0, 0.0, roi),
        roi_to_image_point(float(INPUT_SIZE), 0.0, roi),
        roi_to_image_point(0.0, float(INPUT_SIZE), roi)
    ], dtype=np.float32)

    destination = np.asarray([
        [0.0, 0.0],
        [float(input_w), 0.0],
        [0.0, float(input_h)]
    ], dtype=np.float32)

    transform = cv2.getAffineTransform(source, destination)

    crop = cv2.warpAffine(
        image,
        transform,
        (input_w, input_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb_float = rgb_img.astype(np.float32)

    if tensor_type == 0:
        input_tensor = rgb_float / 255.0
    elif tensor_type in (2, 3, 4):
        inv_scale = np.float32(1.0 / (255.0 * s))
        raw_val = np.round(rgb_float * inv_scale + zp)

        if tensor_type == 2:
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100.0, 100.0)))


def refine_landmarks_from_heatmap(raw_landmarks, heatmap, heatmap_h, heatmap_w, heatmap_channels,
                                  search_radius=HEATMAP_SEARCH_RADIUS,
                                  min_confidence=HEATMAP_MIN_CONFIDENCE):
    refined = raw_landmarks.copy()
    heatmap = np.asarray(heatmap, dtype=np.float32)

    if heatmap.shape != (heatmap_h, heatmap_w, heatmap_channels):
        raise ValueError(
            f"Heatmap data shape {heatmap.shape} does not match "
            f"the supplied shape {(heatmap_h, heatmap_w, heatmap_channels)}"
        )

    if heatmap_channels < raw_landmarks.shape[0]:
        raise ValueError(
            f"Heatmap has {heatmap_channels} channels, but there are "
            f"{raw_landmarks.shape[0]} landmarks"
        )

    for landmark_index in range(raw_landmarks.shape[0]):
        raw_x = raw_landmarks[landmark_index, 0]
        raw_y = raw_landmarks[landmark_index, 1]

        center_col = int(raw_x / INPUT_SIZE * heatmap_w)
        center_row = int(raw_y / INPUT_SIZE * heatmap_h)

        if center_col < 0 or center_col >= heatmap_w or center_row < 0 or center_row >= heatmap_h:
            continue

        begin_col = max(0, center_col - search_radius)
        end_col = min(heatmap_w, center_col + search_radius + 1)
        begin_row = max(0, center_row - search_radius)
        end_row = min(heatmap_h, center_row + search_radius + 1)

        logits = heatmap[begin_row:end_row, begin_col:end_col, landmark_index]
        confidence = sigmoid(logits)

        confidence_sum = float(np.sum(confidence))
        max_confidence = float(np.max(confidence))

        if max_confidence < min_confidence or confidence_sum <= 0.0:
            continue

        columns = np.arange(begin_col, end_col, dtype=np.float32)
        rows = np.arange(begin_row, end_row, dtype=np.float32)

        normalized_x = float(np.sum(confidence * columns[None, :]) / (heatmap_w * confidence_sum))
        normalized_y = float(np.sum(confidence * rows[:, None]) / (heatmap_h * confidence_sum))

        refined[landmark_index, 0] = normalized_x * INPUT_SIZE
        refined[landmark_index, 1] = normalized_y * INPUT_SIZE

    return refined


def postprocess(outputs, roi, image_shape, presence_threshold):
    raw_landmarks = np.asarray(outputs[0], dtype=np.float32).reshape(NUM_LANDMARKS, 5)
    pose_score = float(np.asarray(outputs[1], dtype=np.float32).reshape(-1)[0])
    segmentation = np.asarray(outputs[2], dtype=np.float32)
    heatmap = np.asarray(outputs[3], dtype=np.float32)
    raw_world = np.asarray(outputs[4], dtype=np.float32).reshape(NUM_LANDMARKS, 3)

    if pose_score < presence_threshold:
        return None

    if segmentation.size != INPUT_SIZE * INPUT_SIZE:
        raise ValueError(f"Unexpected segmentation output size: {segmentation.size}")

    if heatmap.ndim == 4:
        if heatmap.shape[0] != 1:
            raise ValueError(f"Expected heatmap batch size 1, got {heatmap.shape}")

        heatmap = heatmap[0]

    if heatmap.ndim != 3:
        raise ValueError(f"Expected NHWC/HWC heatmap, got {heatmap.shape}")

    heatmap_h, heatmap_w, heatmap_channels = heatmap.shape

    refined_landmarks = refine_landmarks_from_heatmap(
        raw_landmarks,
        heatmap,
        heatmap_h,
        heatmap_w,
        heatmap_channels
    )

    height, width = image_shape[:2]
    cosine = math.cos(roi["rotation"])
    sine = math.sin(roi["rotation"])
    landmarks = []

    for i in range(NUM_POSE_LANDMARKS):
        raw = refined_landmarks[i]
        world = raw_world[i]

        x_px, y_px = roi_to_image_point(raw[0], raw[1], roi)
        world_x = cosine * world[0] - sine * world[1]
        world_y = sine * world[0] + cosine * world[1]

        landmarks.append({
            "x": float(x_px / width),
            "y": float(y_px / height),
            "z": float(raw[2] * roi["size"] / (INPUT_SIZE * width)),
            "visibility": float(sigmoid(raw[3])),
            "presence": float(sigmoid(raw[4])),
            "world": [float(world_x), float(world_y), float(world[2])]
        })

    return {
        "score": pose_score,
        "landmarks": landmarks
    }


def draw_detections(image, results, save_path=None, visibility_threshold=0.5, in_place=False):
    result_img = image if in_place else image.copy()
    height, width = result_img.shape[:2]

    for result in results:
        landmarks = result["landmarks"]

        for a, b in SKELETON:
            if landmarks[a]["visibility"] < visibility_threshold or landmarks[b]["visibility"] < visibility_threshold:
                continue

            point_a = (int(landmarks[a]["x"] * width), int(landmarks[a]["y"] * height))
            point_b = (int(landmarks[b]["x"] * width), int(landmarks[b]["y"] * height))
            cv2.line(result_img, point_a, point_b, (0, 255, 0), 2)

        for landmark in landmarks:
            if landmark["visibility"] < visibility_threshold:
                continue

            x = int(landmark["x"] * width)
            y = int(landmark["y"] * height)

            if 0 <= x < width and 0 <= y < height:
                cv2.circle(result_img, (x, y), 3, (0, 0, 255), -1)

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img


def save_landmarks(path, results):
    with open(path, "w", encoding="utf-8") as file:
        for pose_index, result in enumerate(results):
            for landmark_index, landmark in enumerate(result["landmarks"]):
                world = landmark["world"]

                file.write(
                    f"{pose_index} {landmark_index} "
                    f"{landmark['x']:.8f} {landmark['y']:.8f} {landmark['z']:.8f} "
                    f"{landmark['visibility']:.8f} {landmark['presence']:.8f} "
                    f"{world[0]:.8f} {world[1]:.8f} {world[2]:.8f}\n"
                )


def main():
    parser = argparse.ArgumentParser(description="BlazePose Landmark Demo")
    parser.add_argument("--model-path", required=True, help="Path to .adla model")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--detections-dir", required=True, help="Directory containing *_det.txt files")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    tensor_attr = tensor_info["inputs"][0]
    input_h = int(tensor_attr["dims"][1])
    input_w = int(tensor_attr["dims"][2])
    input_shape = (input_h, input_w)
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    if input_shape != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError(f"Expected a {INPUT_SIZE}x{INPUT_SIZE} input, got {input_h}x{input_w}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        amlnn.uninit()
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")

    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")

    print()

    model_name = Path(args.model_path).stem
    result_dir = f"{model_name}_result"
    os.makedirs(result_dir, exist_ok=True)

    for i, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            image = cv2.imread(image_path)

            if image is None:
                raise ValueError(f"can't read image: {image_path}")

            image_name = Path(image_path).stem
            txt_path = os.path.join(args.detections_dir, f"{image_name}_det.txt")
            detections = load_detections(txt_path)

            if not detections:
                print("    No detections found, skipping...")
                print()
                continue

            results = []

            for detection_index, detection in enumerate(detections):
                roi = detection_to_roi(detection, image.shape)
                input_tensor = preprocess(image, roi, input_shape, s, zp, tensor_type)

                outputs = amlnn.inference(inputs=[input_tensor])
                result = postprocess(outputs, roi, image.shape, args.conf)

                if result is not None:
                    results.append(result)

            save_path = os.path.join(result_dir, f"{image_name}_result.jpg")
            landmark_path = os.path.join(result_dir, f"{image_name}_landmarks.txt")

            draw_detections(image, results, save_path)
            save_landmarks(landmark_path, results)

            print(f"    Detected {len(results)} pose(s)")
            print(f"    Result saved to: {save_path}")
            print(f"    Landmarks saved to: {landmark_path}")

        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")

        print()

    print("=" * 60)
    print(amlnn.get_perf_info())

    amlnn.perf_visualize()
    amlnn.uninit()


if __name__ == "__main__":
    main()
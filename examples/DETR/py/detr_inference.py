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

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

class_names = {
    i: name for i, name in enumerate([
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
        "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "doughnut",
        "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
        "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ])
}


def preprocess(img_path, new_shape, s, zp, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    input_h, input_w = new_shape
    resized_img = cv2.resize(original_img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    input_float = rgb_img.astype(np.float32)
    input_float = (input_float - IMAGENET_MEAN) / IMAGENET_STD

    if tensor_type == 0:  # FP32 and FP16 ADLA models use float32 host input
        input_tensor = input_float
    elif tensor_type in (2, 3, 4):
        raw_val = np.round(input_float / s + zp)
        if tensor_type == 2:
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, original_img


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def postprocess(outputs, original_shape, conf_threshold):
    logits_output = np.asarray(outputs[0], dtype=np.float32)
    boxes_output = np.asarray(outputs[1], dtype=np.float32)
    logits = logits_output.reshape(-1, logits_output.shape[-1])
    pred_boxes = boxes_output.reshape(-1, 4)

    if logits.shape[0] != pred_boxes.shape[0]:
        raise ValueError(f"Query count mismatch: logits={logits.shape}, pred_boxes={pred_boxes.shape}")

    probabilities = softmax(logits, axis=1)
    foreground_probabilities = probabilities[:, :-1]
    scores = np.max(foreground_probabilities, axis=1)
    class_ids = np.argmax(foreground_probabilities, axis=1)
    valid_indices = np.where(scores >= conf_threshold)[0]

    original_h, original_w = original_shape[:2]
    detections = []

    for idx in valid_indices:
        class_id = int(class_ids[idx])
        class_name = class_names.get(class_id, f"class_{class_id}")
        if class_name == "N/A":
            continue

        center_x, center_y, width, height = pred_boxes[idx]
        x1 = (center_x - width * 0.5) * original_w
        y1 = (center_y - height * 0.5) * original_h
        x2 = (center_x + width * 0.5) * original_w
        y2 = (center_y + height * 0.5) * original_h

        x1 = float(np.clip(x1, 0, original_w - 1))
        y1 = float(np.clip(y1, 0, original_h - 1))
        x2 = float(np.clip(x2, 0, original_w - 1))
        y2 = float(np.clip(y2, 0, original_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(scores[idx]),
            "class_id": class_id,
            "class_name": class_name
        })

    detections.sort(key=lambda det: det["confidence"], reverse=True)
    return detections


def get_class_color_and_text_color(class_id):
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    text_color = (255, 255, 255) if sum(bgr) < 400 else (0, 0, 0)
    return bgr, text_color


def draw_detections(img, detections, save_path=None, in_place=False):
    result_img = img if in_place else img.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(value) for value in det["bbox"]]
        color, text_color = get_class_color_and_text_color(det["class_id"])
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        y1_label = max(y1, label_h + 10)

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(result_img, (x1, y1_label - label_h - 10), (x1 + label_w, y1_label), color, thickness=cv2.FILLED)
        cv2.putText(result_img, label, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness=1, lineType=cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img


def main():
    parser = argparse.ArgumentParser(description="DETR Demo")
    parser.add_argument("--adla", required=True, help="Path to .adla model")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.adla)
    tensor_info = amlnn.get_tensor_info()
    print(amlnn.get_sdk_version())

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files = sorted(image_files)

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        amlnn.uninit()
        return 0

    tensor_attr = tensor_info["inputs"][0]
    input_h = int(tensor_attr["dims"][1])
    input_w = int(tensor_attr["dims"][2])
    input_shape = (input_h, input_w)
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    model_name = Path(args.adla).stem
    result_dir = f"{model_name}_result"
    os.makedirs(result_dir, exist_ok=True)

    print(f"Input shape (NHWC): {tensor_attr['dims']}")
    print(f"Found {len(image_files)} image file(s) to process")

    for i, image_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")

        try:
            input_tensor, original_img = preprocess(image_path, input_shape, s, zp, tensor_type)
            outputs = amlnn.inference(inputs=[input_tensor])
            detections = postprocess(outputs, original_img.shape, args.conf)

            if detections:
                print(f"Detected {len(detections)} objects:")
                for j, det in enumerate(detections, 1):
                    print(f"  {j}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("No objects detected")

            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")
            draw_detections(original_img, detections, save_path)
            print(f"Result saved to: {save_path}")
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

    print(amlnn.get_perf_info())
    # amlnn.perf_visualize()
    amlnn.uninit()
    return 0


if __name__ == "__main__":
    main()
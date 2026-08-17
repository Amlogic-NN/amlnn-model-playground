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

REG_MAX = 16
class_names = [
    'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
    'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt',
    'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']

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

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

def postprocess(outputs, input_shape, scale, pad, conf_threshold, iou_threshold, reg_max = REG_MAX):
    input_h, input_w = input_shape
    dfl_channels = 4 * reg_max
    safe_threshold = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inverse_threshold = np.log(safe_threshold / (1.0 - safe_threshold))
    projection = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max)
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Combined NHWC heads contain 64 DFL logits followed by 13 class logits.
    strides = [8, 16, 32]
    for output_idx, stride in enumerate(strides):
        output = np.asarray(outputs[output_idx])
        expected_h = input_h // stride
        expected_w = input_w // stride
        expected_channels = dfl_channels + len(class_names)
        expected_shape = (1, expected_h, expected_w, expected_channels)
        if output.shape != expected_shape:
            raise ValueError(
                f"Unexpected output shape for stride {stride}: {output.shape}; "
                f"expected {expected_shape}"
            )

        height = output.shape[1]
        width = output.shape[2]
        predictions = output.reshape(height * width, expected_channels)
        class_logits = predictions[:, dfl_channels:]
        max_raw_scores = np.max(class_logits, axis=1)
        valid_indices = np.where(max_raw_scores > inverse_threshold)[0]
        if len(valid_indices) == 0:
            continue

        valid_predictions = predictions[valid_indices]
        valid_class_logits = class_logits[valid_indices]
        scores = sigmoid(max_raw_scores[valid_indices])
        class_ids = np.argmax(valid_class_logits, axis=1)

        # Decode four 16-bin distributions into left, top, right, and bottom distances.
        dfl_logits = valid_predictions[:, :dfl_channels].reshape(-1, 4, reg_max)
        dfl_logits -= np.max(dfl_logits, axis=2, keepdims=True)
        probabilities = np.exp(dfl_logits)
        probabilities /= np.sum(probabilities, axis=2, keepdims=True)
        distances = np.sum(probabilities * projection, axis=2)

        grid_x = (valid_indices % width).astype(np.float32)
        grid_y = (valid_indices // width).astype(np.float32)
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

    # Undo letterbox padding and resize scaling independently on both axes.
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes = np.maximum(boxes, 0.0)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    boxes_xywh = np.stack([boxes[:, 0], boxes[:, 1], widths, heights], axis=1)

    # Run NMS separately for each prompted class.
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
            'bbox': boxes[detection_idx].tolist(),
            'confidence': float(scores[detection_idx]),
            'class_id': class_id,
            'class_name': class_names[class_id]
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
    parser = argparse.ArgumentParser(description="YOLOWorld Demo")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms', type=float, default=0.45)
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.adla)
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
            detections = postprocess(
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

            result_dir = f"{Path(args.adla).stem}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(original_img, detections, save_path)
            print(f"    Result saved to: {save_path}")
        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")

        print()

    print("=" * 60)
    print(amlnn.get_perf_info())
    # amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
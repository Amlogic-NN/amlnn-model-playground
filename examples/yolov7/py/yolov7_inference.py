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

# Standard YOLOv7 anchors for strides 8, 16, and 32.
ANCHORS = {
    8: np.array([[12, 16], [19, 36], [40, 28]], dtype=np.float32),
    16: np.array([[36, 75], [76, 55], [72, 146]], dtype=np.float32),
    32: np.array([[142, 110], [192, 243], [459, 401]], dtype=np.float32)
}

class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'doughnut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
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

def preprocess(img_path, new_shape, s, zp, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    rgb_float = rgb_img.astype(np.float32)

    # Normalize & Quantization
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
    return input_tensor, original_img, scale, pad

def postprocess(outputs, input_shape, scale, pad, conf_threshold, iou_threshold):
    input_h, input_w = input_shape
    all_boxes = []
    all_scores = []
    all_class_ids = []

    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))

    # Output order: [stride 8, stride 16, stride 32], each as NHWC [1, H, W, 255].
    strides = [8, 16, 32]

    for output_idx, stride in enumerate(strides):
        output = np.asarray(outputs[output_idx])
        if output.ndim != 4:
            raise ValueError(f"Unexpected output shape for stride {stride}: {output.shape}")

        batch_size, height, width, channels = output.shape

        num_anchors = len(ANCHORS[stride])
        if channels % num_anchors != 0:
            raise ValueError(f"Output channel count {channels} is not divisible by {num_anchors} anchors")

        values_per_anchor = channels // num_anchors
        num_classes = values_per_anchor - 5
        if num_classes <= 0:
            raise ValueError(f"Invalid YOLOv7 channel count for stride {stride}: {channels}")

        predictions = output.reshape(height * width, num_anchors, values_per_anchor)
        objectness_logits = predictions[..., 4]
        valid_mask = objectness_logits > inv_thresh
        if not np.any(valid_mask):
            continue

        # Decode only anchors whose objectness can still pass the final confidence threshold.
        valid_predictions = predictions[valid_mask]
        grid_indices, anchor_indices = np.where(valid_mask)
        valid_predictions = 1.0 / (1.0 + np.exp(-np.clip(valid_predictions, -80.0, 80.0)))

        class_probabilities = valid_predictions[:, 5:]
        class_ids = np.argmax(class_probabilities, axis=1)
        scores = valid_predictions[:, 4] * np.max(class_probabilities, axis=1)
        score_mask = scores > conf_threshold
        if not np.any(score_mask):
            continue

        valid_predictions = valid_predictions[score_mask]
        scores = scores[score_mask]
        class_ids = class_ids[score_mask]
        grid_indices = grid_indices[score_mask]
        anchor_indices = anchor_indices[score_mask]

        grid_x = (grid_indices % width).astype(np.float32)
        grid_y = (grid_indices // width).astype(np.float32)
        grid = np.stack([grid_x, grid_y], axis=1)
        anchors = ANCHORS[stride][anchor_indices]

        # Standard YOLOv7 anchor decode from sigmoid tx, ty, tw, and th values.
        centers = (valid_predictions[:, 0:2] * 2.0 - 0.5 + grid) * stride
        sizes = (valid_predictions[:, 2:4] * 2.0) ** 2 * anchors
        x1 = centers[:, 0] - sizes[:, 0] / 2.0
        y1 = centers[:, 1] - sizes[:, 1] / 2.0
        x2 = centers[:, 0] + sizes[:, 0] / 2.0
        y2 = centers[:, 1] + sizes[:, 1] / 2.0

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores)
        all_class_ids.append(class_ids)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    # Undo letterbox scaling, then use class offsets for per-class OpenCV NMS.
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes = np.maximum(boxes, 0.0)

    max_coord = np.max(boxes) + 1.0
    offsets = class_ids.astype(boxes.dtype) * max_coord
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    boxes_xywh = np.stack([boxes[:, 0] + offsets, boxes[:, 1] + offsets, widths, heights], axis=1)
    nms_indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_threshold, iou_threshold)

    detections = []
    if len(nms_indices) > 0:
        for detection_idx in nms_indices.flatten():
            x1, y1, x2, y2 = boxes[detection_idx]
            class_id = int(class_ids[detection_idx])
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(scores[detection_idx]),
                'class_id': class_id,
                'class_name': class_names.get(class_id, f'class_{class_id}')
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

    for det in detections:
        x1, y1, x2, y2 = [int(value) for value in det['bbox']]
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']
        color, text_color = get_class_color_and_text_color(class_id)

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = max(y1, label_h + 10)
        cv2.rectangle(result_img, (x1, label_y - label_h - 10), (x1 + label_w, label_y), color, cv2.FILLED)
        cv2.putText(
            result_img, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, text_color, thickness=1, lineType=cv2.LINE_AA
        )

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img

def main():
    parser = argparse.ArgumentParser(description="YOLOv7 Demo")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--nms', type=float, default=0.45)
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.adla)
    tensor_info = amlnn.get_tensor_info()
    print(amlnn.get_sdk_version())

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

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
    input_zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    for image_idx, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {image_idx}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            input_tensor, original_img, scale, pad = preprocess(
                image_path, input_shape, input_scale, input_zp, tensor_type
            )
            outputs = amlnn.inference(inputs=[input_tensor])
            detections = postprocess(outputs, input_shape, scale, pad, args.conf, args.nms)

            if detections:
                print(f"    Detected {len(detections)} objects:")
                for detection_idx, detection in enumerate(detections, 1):
                    print(f"      {detection_idx}. {detection['class_name']} ({detection['confidence']:.2f})")
            else:
                print("    No objects detected")

            result_dir = f"{Path(args.adla).stem}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(original_img, detections, save_path)
            print(f"    Result saved to: {save_path}")
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print("=" * 60)
    print(amlnn.get_perf_info())
    # amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
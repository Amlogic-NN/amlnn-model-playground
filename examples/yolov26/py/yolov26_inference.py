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

MAX_DETECTIONS = 300

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

    # 3. Fused Normalization & Quantization if needed
    if tensor_type == 0: # FP32 & FP16
        input_tensor = (rgb_float / 255.0) # Only normalize
    elif tensor_type in (2, 3, 4):
        inv_scale = np.float32(1.0 / (255.0 * s))
        raw_val = np.round((rgb_float * inv_scale) + zp)

        if tensor_type == 2:    # Int8
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:  # Uint8
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:                   # Int16
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, original_img, scale, pad

def postprocess(outputs, input_shape, scale, pad, conf_threshold, max_detections=MAX_DETECTIONS):
    # Output order: cls_8, bbox_8, cls_16, bbox_16, cls_32, bbox_32
    input_h, input_w = input_shape

    if len(outputs) != 6:
        raise RuntimeError(f"Expected 6 YOLO26 outputs, got {len(outputs)}")

    all_class_preds = []
    all_bbox_preds = []
    all_anchor_x = []
    all_anchor_y = []
    all_strides = []

    strides = [8, 16, 32]

    for scale_idx, stride in enumerate(strides):
        cls_output = np.asarray(outputs[scale_idx * 2], dtype=np.float32)
        bbox_output = np.asarray(outputs[scale_idx * 2 + 1], dtype=np.float32)

        # AMLNN outputs are NHWC:
        # cls:  [1, H, W, num_classes]
        # bbox: [1, H, W, 4] containing left, top, right, bottom distances
        if cls_output.ndim != 4 or cls_output.shape[0] != 1:
            raise RuntimeError(
                f"Expected class output {scale_idx * 2} shape [1, H, W, C], "
                f"got {cls_output.shape}"
            )

        if bbox_output.ndim != 4 or bbox_output.shape[0] != 1:
            raise RuntimeError(
                f"Expected bbox output {scale_idx * 2 + 1} shape [1, H, W, 4], "
                f"got {bbox_output.shape}"
            )

        _, height, width, num_classes = cls_output.shape
        _, bbox_height, bbox_width, bbox_channels = bbox_output.shape

        if bbox_height != height or bbox_width != width or bbox_channels != 4:
            raise RuntimeError(
                f"Bbox output {scale_idx * 2 + 1} shape {bbox_output.shape} does not match "
                f"expected [1, {height}, {width}, 4]"
            )

        expected_height = input_h // stride
        expected_width = input_w // stride
        if height != expected_height or width != expected_width:
            raise RuntimeError(
                f"Stride {stride} output grid is {height}x{width}, "
                f"expected {expected_height}x{expected_width}"
            )

        class_preds = cls_output.reshape(-1, num_classes)
        bbox_preds = bbox_output.reshape(-1, 4)

        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )

        all_class_preds.append(class_preds)
        all_bbox_preds.append(bbox_preds)
        all_anchor_x.append((grid_x.reshape(-1) + 0.5).astype(np.float32))
        all_anchor_y.append((grid_y.reshape(-1) + 0.5).astype(np.float32))
        all_strides.append(np.full(height * width, stride, dtype=np.float32))

    class_preds = np.concatenate(all_class_preds, axis=0)
    bbox_preds = np.concatenate(all_bbox_preds, axis=0)
    anchor_x = np.concatenate(all_anchor_x, axis=0)
    anchor_y = np.concatenate(all_anchor_y, axis=0)
    stride_values = np.concatenate(all_strides, axis=0)

    num_locations, num_classes = class_preds.shape
    location_k = min(max_detections, num_locations)

    # Match the one-to-one head's two-stage Top-K selection. Sigmoid is
    # monotonic, so ranking raw logits gives the same ordering while avoiding
    # sigmoid over every class score.
    location_scores = np.max(class_preds, axis=1)
    if location_k < num_locations:
        top_location_indices = np.argpartition(
            location_scores,
            num_locations - location_k,
        )[-location_k:]
    else:
        top_location_indices = np.arange(num_locations)

    selected_class_preds = class_preds[top_location_indices]
    flat_class_preds = selected_class_preds.reshape(-1)
    pair_k = min(max_detections, flat_class_preds.size)

    if pair_k < flat_class_preds.size:
        top_pair_indices = np.argpartition(
            flat_class_preds,
            flat_class_preds.size - pair_k,
        )[-pair_k:]
    else:
        top_pair_indices = np.arange(flat_class_preds.size)

    top_pair_indices = top_pair_indices[
        np.argsort(flat_class_preds[top_pair_indices])[::-1]
    ]

    selected_location_positions = top_pair_indices // num_classes
    class_ids = (top_pair_indices % num_classes).astype(np.int32)
    location_indices = top_location_indices[selected_location_positions]
    raw_scores = flat_class_preds[top_pair_indices]

    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))
    valid_mask = raw_scores > inv_thresh

    if not np.any(valid_mask):
        return []

    location_indices = location_indices[valid_mask]
    class_ids = class_ids[valid_mask]
    raw_scores = raw_scores[valid_mask]
    scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -80.0, 80.0)))

    # YOLO26 has no DFL softmax/expected-value decode, but the four direct
    # left/top/right/bottom distances still require anchor/grid decoding.
    selected_bbox_preds = bbox_preds[location_indices]
    selected_anchor_x = anchor_x[location_indices]
    selected_anchor_y = anchor_y[location_indices]
    selected_strides = stride_values[location_indices]

    left, top, right, bottom = selected_bbox_preds.T
    x1 = (selected_anchor_x - left) * selected_strides
    y1 = (selected_anchor_y - top) * selected_strides
    x2 = (selected_anchor_x + right) * selected_strides
    y2 = (selected_anchor_y + bottom) * selected_strides

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Map coordinates back to the original image.
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes = np.maximum(boxes, 0)

    detections = []
    for index in range(len(scores)):
        bx1, by1, bx2, by2 = boxes[index]
        class_id = int(class_ids[index])
        detections.append({
            'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
            'confidence': float(scores[index]),
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
        bbox = det['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']

        color, text_color = get_class_color_and_text_color(class_id)

        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # 5. Prevent label from drawing outside the top image boundary
        y1_label = max(y1, label_h + 10)

        # Draw background rectangle for label
        cv2.rectangle(
            result_img,
            (x1, y1_label - label_h - 10),
            (x1 + label_w, y1_label),
            color,
            thickness=cv2.FILLED
        )

        # Draw text
        cv2.putText(
            result_img,
            label,
            (x1, y1_label - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img

def main():
    parser = argparse.ArgumentParser(description="YOLO26 Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()

    if len(tensor_info["outputs"]) != 6:
        raise RuntimeError(f"Expected 6 YOLO26 outputs, got {len(tensor_info['outputs'])}")

    print(amlnn.get_sdk_version())

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        amlnn.uninit()
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    tensor_attr = tensor_info["inputs"][0]
    input_h = int(tensor_attr["dims"][1])
    input_w = int(tensor_attr["dims"][2])
    input_shape = (input_h, input_w)
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess input
            input_tensor, original_img, scale, pad = preprocess(
                image_path, input_shape, s, zp, tensor_type
            )

            # Run inference
            outputs = amlnn.inference(
                inputs=[input_tensor]
            )

            # Postprocess one-to-one head results
            detections = postprocess(
                outputs, input_shape, scale, pad, args.conf
            )

            # Print detection results
            if detections:
                print(f"    Detected {len(detections)} objects:")
                for detection_index, det in enumerate(detections, 1):
                    print(
                        f"      {detection_index}. "
                        f"{det['class_name']} ({det['confidence']:.2f})"
                    )
            else:
                print("    No objects detected")

            # Save result image
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

    print(f"=" * 60)
    print(amlnn.get_perf_info())

    # Release resources
    amlnn.uninit()

if __name__ == "__main__":
    main()
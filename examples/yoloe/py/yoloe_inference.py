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

def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: name for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not load class names from '{path}'. Fallback to generic IDs.")
        return {}

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

def postprocess(outputs, input_shape, scale, pad, conf_threshold, iou_threshold, class_names, regmax=16):
    # input_shape is a tuple: (input_height, input_width)
    input_h, input_w = input_shape

    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Calculate inverse sigmoid threshold for early stopping
    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))

    # NOTE: You must ensure `outputs` are ordered [DFL_32, CLS_32, DFL_16, CLS_16, DFL_8, CLS_8]
    strides = [32, 16, 8]

    for idx, stride in enumerate(strides):
        dfl_out = outputs[idx * 2]
        cls_out = outputs[idx * 2 + 1]

        dfl_sq = np.squeeze(dfl_out)
        cls_sq = np.squeeze(cls_out)

        if dfl_sq.shape[0] < dfl_sq.shape[1]:
            dfl_preds = dfl_sq.T
            class_preds = cls_sq.T
        else:
            dfl_preds = dfl_sq
            class_preds = cls_sq

        # Grid width based on input width and current stride
        width = input_w // stride

        regression_range = np.arange(regmax, dtype=np.float32)

        # 2. Early filtering
        max_raw_scores = np.max(class_preds, axis=1)
        valid_mask = max_raw_scores > inv_thresh

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        valid_class_preds = class_preds[valid_indices]
        valid_dfl_preds = dfl_preds[valid_indices]

        # Apply sigmoid to valid scores
        valid_class_scores = 1.0 / (1.0 + np.exp(-valid_class_preds))
        max_class_scores = np.max(valid_class_scores, axis=1)
        class_ids = np.argmax(valid_class_scores, axis=1)

        # 3. Grid generation
        grid_y = (valid_indices // width).astype(np.float32)
        grid_x = (valid_indices % width).astype(np.float32)

        # 4. DFL decoding
        dfl_reshaped = valid_dfl_preds.reshape(-1, 4, regmax)
        dfl_max = np.max(dfl_reshaped, axis=-1, keepdims=True)
        exp_dfl = np.exp(dfl_reshaped - dfl_max)
        dfl_softmax = exp_dfl / np.sum(exp_dfl, axis=-1, keepdims=True)
        bbox_deltas = np.sum(dfl_softmax * regression_range[None, None, :], axis=-1)

        # 5. Absolute coordinates
        anchor_x = (grid_x + 0.5) * stride
        anchor_y = (grid_y + 0.5) * stride

        left, top, right, bottom = bbox_deltas.T
        x1 = anchor_x - left * stride
        y1 = anchor_y - top * stride
        x2 = anchor_x + right * stride
        y2 = anchor_y + bottom * stride

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        all_boxes.append(boxes)
        all_scores.append(max_class_scores)
        all_class_ids.append(class_ids)

    # Merge all scales
    if not all_boxes:
        return []

    valid_boxes = np.concatenate(all_boxes, axis=0)
    valid_scores = np.concatenate(all_scores, axis=0)
    valid_class_ids = np.concatenate(all_class_ids, axis=0)

    # Map coordinates back to original image scaling
    pad_x, pad_y = pad
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale

    valid_boxes = np.maximum(valid_boxes, 0)

    # Safe class name getter
    def get_class_name(cid):
        if isinstance(class_names, dict):
            return class_names.get(cid, f'class_{cid}')
        elif isinstance(class_names, (list, tuple)):
            if 0 <= cid < len(class_names):
                return class_names[cid]
        return f'class_{cid}'

    detections = []

    # 6. Per-Class NMS
    unique_classes = np.unique(valid_class_ids)

    for c in unique_classes:
        class_mask = valid_class_ids == c
        c_boxes = valid_boxes[class_mask]
        c_scores = valid_scores[class_mask]

        # NMSBoxes needs [x1, y1, w, h]
        c_widths = c_boxes[:, 2] - c_boxes[:, 0]
        c_heights = c_boxes[:, 3] - c_boxes[:, 1]
        c_boxes_xywh = np.stack([c_boxes[:, 0], c_boxes[:, 1], c_widths, c_heights], axis=1)

        nms_indices = cv2.dnn.NMSBoxes(
            c_boxes_xywh.tolist(), c_scores.tolist(), conf_threshold, iou_threshold
        )

        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            for idx in nms_indices:
                bx1, by1, bx2, by2 = c_boxes[idx]
                detections.append({
                    'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                    'confidence': float(c_scores[idx]),
                    'class_id': int(c),
                    'class_name': get_class_name(int(c))
                })

    return detections

def get_class_color_and_text_color(class_id):
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    # Precompute text color based on background brightness
    text_color = (255, 255, 255) if sum(bgr) < 400 else (0, 0, 0)
    return bgr, text_color


def draw_detections(img, detections, save_path):
    result_img = img.copy()
    img_h, img_w = result_img.shape[:2]
    print(f"    Detected {len(detections)} objects")

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = map(int, det['bbox'])

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        confidence = det['confidence']
        class_name = det['class_name']

        print(f"      {i}. {class_name} ({confidence:.2f}) -> [{x1}, {y1}, {x2}, {y2}]")

        color, text_color = get_class_color_and_text_color(det['class_id'])

        # Draw the bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # 1. Prevent label from going off the Right edge
        label_x = x1
        if label_x + label_w > img_w:
            label_x = img_w - label_w

        # 2. Prevent label from going off the Top edge
        label_top_y = y1 - label_h - 10
        label_bottom_y = y1
        text_y = y1 - 5

        # If it goes off the top screen, flip the label to be INSIDE the bounding box
        if label_top_y < 0:
            label_top_y = y1
            label_bottom_y = y1 + label_h + 10
            text_y = y1 + label_h + 5

        # Draw label background
        cv2.rectangle(result_img, (label_x, label_top_y), (label_x + label_w, label_bottom_y), color, -1)

        # Draw text
        cv2.putText(result_img, label, (label_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(save_path, result_img)

def main():
    parser = argparse.ArgumentParser(description="Yoloe Demo")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--labels', default="../input/labels.txt", help='Path of the labels.txt')
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--nms", type=float, default=0.05)
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

    if not image_files:
        print(f"No image files found in {args.image_dir}")
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

    class_names = load_class_names(args.labels)

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            input_tensor, original_img, scale, pad = preprocess(image_path, input_shape, s, zp, tensor_type)

            outputs = amlnn.inference(inputs=[input_tensor])

            detections = postprocess(outputs, input_shape, scale, pad, args.conf, args.nms, class_names)

            if detections:
                print(f"Detected {len(detections)} objects in {os.path.basename(image_path)}")
            else:
                print(f"No objects detected in {os.path.basename(image_path)}")

            model_name = Path(args.adla).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")
            draw_detections(original_img, detections, str(save_path))
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
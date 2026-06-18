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
from pathlib import Path
from amlnn.api import AMLNN

MEAN = np.array([0, 0, 0], dtype=np.float32)
STD  = np.array([255, 255, 255], dtype=np.float32)

def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: name for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not load class names from '{path}'. Fallback to generic IDs.")
        return {}

CLASS_NAMES = load_class_names("../input/coco_80_names.txt")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, scale, (left, top)

def preprocess(img_path, new_shape=(640,640), s=0.003850, zp=-128, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None: return None, None, None, None

    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    normalized_img = (rgb_img.astype(np.float32) - MEAN) / STD

    input_tensor = np.expand_dims(normalized_img, axis=0)

    raw_val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)

    return input_tensor, original_img, scale, pad

def postprocess(outputs, scale, pad, class_names, conf_threshold=0.25, iou_threshold=0.45):
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Inverse sigmoid for early stopping on raw logits
    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))

    strides = [8, 16, 32]

    for s_idx in range(3):
        # s_idx 0 -> cls=0 (Reshape_3), dfl=1 (Reshape_0)
        # s_idx 1 -> cls=2 (Reshape_4), dfl=3 (Reshape_1)
        # s_idx 2 -> cls=4 (Reshape_5), dfl=5 (Reshape_2)
        cls_idx = s_idx * 2
        dfl_idx = s_idx * 2 + 1

        stride = strides[s_idx]
        grid_size = 640 // stride
        num_cells = grid_size * grid_size  # 6400, 1600, or 400

        cls_raw = np.squeeze(outputs[cls_idx])
        dfl_raw = np.squeeze(outputs[dfl_idx])

        if cls_raw.shape[1] != num_cells:
            cls_raw = cls_raw.T
        if dfl_raw.shape[1] != num_cells:
            dfl_raw = dfl_raw.T

        # Find max logit score across classes
        max_raw_scores = np.max(cls_raw, axis=0)

        # Early Stopping check against inverse threshold
        valid_mask = max_raw_scores > inv_thresh
        if not np.any(valid_mask):
            continue

        valid_cls_ids = np.argmax(cls_raw[:, valid_mask], axis=0)
        valid_raw_scores = max_raw_scores[valid_mask]

        # Convert valid raw logits to probabilities via Sigmoid
        valid_scores = 1.0 / (1.0 + np.exp(-valid_raw_scores))

        # Extract valid DFL predictions
        valid_dfl_raw = dfl_raw[:, valid_mask] # (64, V)
        V = valid_dfl_raw.shape[1]

        # 16-Bin DFL Decode
        dfl = valid_dfl_raw.reshape(4, 16, V)
        dfl_max = np.max(dfl, axis=1, keepdims=True)
        exp_dfl = np.exp(dfl - dfl_max)
        softmax_dfl = exp_dfl / np.sum(exp_dfl, axis=1, keepdims=True)

        weights = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
        ltrb = np.sum(softmax_dfl * weights, axis=1) # (4, V)

        # Generate grid coordinates
        grid_y, grid_x = np.mgrid[0:grid_size, 0:grid_size]
        valid_gx = grid_x.flatten()[valid_mask]
        valid_gy = grid_y.flatten()[valid_mask]

        cx = (valid_gx + 0.5) * stride
        cy = (valid_gy + 0.5) * stride

        x1 = cx - ltrb[0] * stride
        y1 = cy - ltrb[1] * stride
        x2 = cx + ltrb[2] * stride
        y2 = cy + ltrb[3] * stride

        valid_boxes = np.stack([x1, y1, x2, y2], axis=1)

        all_boxes.append(valid_boxes)
        all_scores.append(valid_scores)
        all_class_ids.append(valid_cls_ids)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    # Revert Letterbox Scale and Pad
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes = np.maximum(boxes, 0)

    detections = []

    # EXACT PER-CLASS NMS
    unique_classes = np.unique(class_ids)

    for c in unique_classes:
        class_mask = class_ids == c
        c_boxes = boxes[class_mask]
        c_scores = scores[class_mask]

        # NMSBoxes needs [x, y, w, h] format
        c_widths = c_boxes[:, 2] - c_boxes[:, 0]
        c_heights = c_boxes[:, 3] - c_boxes[:, 1]
        c_boxes_xywh = np.stack([c_boxes[:, 0], c_boxes[:, 1], c_widths, c_heights], axis=1)

        nms_indices = cv2.dnn.NMSBoxes(
            c_boxes_xywh.tolist(), c_scores.tolist(), conf_threshold, iou_threshold
        )

        if len(nms_indices) > 0:
            nms_indices = np.array(nms_indices).flatten()
            for idx in nms_indices:
                bx1, by1, bx2, by2 = c_boxes[idx]
                detections.append({
                    'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                    'confidence': float(c_scores[idx]),
                    'class_id': int(c),
                    'class_name': class_names.get(int(c), f'class_{int(c)}')
                })

    return detections

def get_class_color(class_id):
    import colorsys
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    return (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))

def draw_detections(img, detections, save_path):
    result_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        color = get_class_color(det['class_id'])

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(save_path, result_img)

def main():
    parser = argparse.ArgumentParser(description="Yolov10 Demo")
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
        try:
            input_tensor, original_img, scale, pad = preprocess(
                image_path, new_shape=(640, 640), s=s, zp=zp, tensor_type=tensor_type
            )

            outputs = amlnn.inference(inputs=[input_tensor])

            detections = postprocess(outputs, scale, pad, CLASS_NAMES, conf_threshold=0.25, iou_threshold=0.45)

            if detections:
                print(f"Detected {len(detections)} objects in {os.path.basename(image_path)}")
            else:
                print(f"No objects detected in {os.path.basename(image_path)}")

            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")

            draw_detections(original_img, detections, str(save_path))

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        print()
    print(f"=" * 60)
    amlnn.uninit()

if __name__ == "__main__":
    main()
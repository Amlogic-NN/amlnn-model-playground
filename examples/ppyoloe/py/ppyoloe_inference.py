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

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.1

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

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
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

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=0.003789, zp=-128, tensor_type=2):
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
        raise ValueError(f"Unsupported format: {data_format}")

    raw_val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)

    return input_tensor, original_img, scale, pad

def postprocess(outputs, scale, pad, strides=[8, 16, 32], conf_threshold=0.25, iou_threshold=0.45):
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Inverse sigmoid for raw logit thresholding
    safe_thresh = np.clip(conf_threshold, 1e-5, 1.0 - 1e-5)
    inv_thresh = np.log(safe_thresh / (1.0 - safe_thresh))

    num_classes = 80
    num_bins = 17

    # Loop over the 3 stride outputs
    for s_idx, stride in enumerate(strides):
        dfl_out = np.squeeze(outputs[s_idx * 2])
        cls_out = np.squeeze(outputs[s_idx * 2 + 1])

        grid_size = 640 // stride

        # 1. Flatten CLS tensor to (N, 80)
        cls_raw = cls_out.reshape(-1, num_classes) # Shape: (N, 80)

        # 2. Find max confidence for each anchor and threshold
        max_raw_scores = np.max(cls_raw, axis=1) # Shape: (N,)
        valid_mask = max_raw_scores > inv_thresh # Shape: (N,)

        # Early Stopping if no objects found in this stride
        if not np.any(valid_mask):
            continue

        # Filter Classes
        valid_cls_ids = np.argmax(cls_raw[valid_mask, :], axis=1) # (V,)
        valid_raw_scores = max_raw_scores[valid_mask]             # (V,)

        # Apply Sigmoid ONLY to valid scores (saves massive CPU time)
        valid_scores = 1.0 / (1.0 + np.exp(-valid_raw_scores))

        # 3. Filter DFL Tensors
        valid_dfl = dfl_out[valid_mask, :, :] # Shape: (V, 17, 4)

        # 4. 17-Bin DFL Decode
        dfl_max = np.max(valid_dfl, axis=1, keepdims=True) # (V, 1, 4)
        exp_dfl = np.exp(valid_dfl - dfl_max)              # (V, 17, 4)
        softmax_dfl = exp_dfl / np.sum(exp_dfl, axis=1, keepdims=True) # (V, 17, 4)

        # Multiply probabilities by [0, 1, 2 ... 16]
        weights = np.arange(num_bins, dtype=np.float32).reshape(1, 17, 1)
        ltrb = np.sum(softmax_dfl * weights, axis=1) # Result shape: (V, 4)

        # 5. Generate grid coordinates for the valid boxes
        gy, gx = np.mgrid[0:grid_size, 0:grid_size]
        valid_gx = gx.flatten()[valid_mask] # (V,)
        valid_gy = gy.flatten()[valid_mask] # (V,)

        cx = (valid_gx + 0.5) * stride
        cy = (valid_gy + 0.5) * stride

        # ltrb is (V, 4). Columns are Left(0), Top(1), Right(2), Bottom(3)
        x1 = cx - ltrb[:, 0] * stride
        y1 = cy - ltrb[:, 1] * stride
        x2 = cx + ltrb[:, 2] * stride
        y2 = cy + ltrb[:, 3] * stride

        valid_boxes = np.stack([x1, y1, x2, y2], axis=1) # Shape: (V, 4)

        all_boxes.append(valid_boxes)
        all_scores.append(valid_scores)
        all_class_ids.append(valid_cls_ids)

    # Merge all scales
    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)

    # 6. Map coordinates back to original image
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes = np.maximum(boxes, 0)

    detections = []

    # 7. EXACT PER-CLASS NMS
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
            nms_indices = nms_indices.flatten()
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
    print(f"    Detected {len(detections)} objects")

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = map(int, det['bbox'])
        confidence = det['confidence']
        class_name = det['class_name']

        print(f"      {i}. {class_name} ({confidence:.2f}) -> [{x1}, {y1}, {x2}, {y2}]")
        color = get_class_color(det['class_id'])
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(save_path, result_img)
    print(f"    Result saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="PPYoloe Demo")
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

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        try:
            input_tensor, original_img, scale, pad = preprocess(
                image_path, new_shape=(640, 640), data_format='NHWC', s=s, zp=zp, tensor_type=tensor_type
            )

            outputs = amlnn.inference(inputs=[input_tensor])

            detections = postprocess(outputs, scale, pad, conf_threshold=CONF_THRESHOLD, iou_threshold=NMS_THRESHOLD)

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
    print("=" * 60)

    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
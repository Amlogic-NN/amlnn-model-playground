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

    if tensor_type == 0:
        input_tensor = rgb_float / 255.0
    elif tensor_type in (2, 3, 4):
        inv_scale = np.float32(1.0 / (255.0 * s))
        raw_val = np.round((rgb_float * inv_scale) + zp)

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

    if len(outputs) < 6:
        raise ValueError(f"Expected 6 YOLOv6 outputs, got {len(outputs)}")

    # Output order: [CLS_8, BBOX_8, CLS_16, BBOX_16, CLS_32, BBOX_32]
    strides = [8, 16, 32]
    for idx, stride in enumerate(strides):
        cls_out = outputs[idx * 2]
        bbox_out = outputs[idx * 2 + 1]

        if cls_out.ndim != 4:
            raise ValueError(f"Unexpected class output shape for stride {stride}: {cls_out.shape}")

        batch_size, height, width, num_classes = cls_out.shape

        class_preds = cls_out.reshape(-1, num_classes)
        bbox_sq = np.squeeze(bbox_out)

        if bbox_sq.shape == (4, height * width):
            bbox_preds = bbox_sq.T
        elif bbox_sq.shape == (height * width, 4):
            bbox_preds = bbox_sq
        else:
            raise ValueError(f"BBox output {bbox_out.shape} does not match the {height}x{width} class grid")

        max_raw_scores = np.max(class_preds, axis=1)
        valid_mask = max_raw_scores > inv_thresh
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        valid_class_preds = class_preds[valid_indices]
        valid_bbox_preds = bbox_preds[valid_indices]
        valid_scores = 1.0 / (1.0 + np.exp(-max_raw_scores[valid_indices]))
        valid_class_ids = np.argmax(valid_class_preds, axis=1)

        grid_y = (valid_indices // width).astype(np.float32)
        grid_x = (valid_indices % width).astype(np.float32)
        center_x = (grid_x + 0.5) * stride
        center_y = (grid_y + 0.5) * stride

        left = valid_bbox_preds[:, 0]
        top = valid_bbox_preds[:, 1]
        right = valid_bbox_preds[:, 2]
        bottom = valid_bbox_preds[:, 3]
        x1 = center_x - left * stride
        y1 = center_y - top * stride
        x2 = center_x + right * stride
        y2 = center_y + bottom * stride
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        all_boxes.append(boxes)
        all_scores.append(valid_scores)
        all_class_ids.append(valid_class_ids)

    if not all_boxes:
        return []

    valid_boxes = np.concatenate(all_boxes, axis=0)
    valid_scores = np.concatenate(all_scores, axis=0)
    valid_class_ids = np.concatenate(all_class_ids, axis=0)

    pad_x, pad_y = pad
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale
    valid_boxes = np.maximum(valid_boxes, 0)

    max_coord = np.max(valid_boxes) + 1.0
    offsets = valid_class_ids.astype(valid_boxes.dtype) * max_coord
    widths = valid_boxes[:, 2] - valid_boxes[:, 0]
    heights = valid_boxes[:, 3] - valid_boxes[:, 1]
    boxes_xywh = np.stack([valid_boxes[:, 0] + offsets, valid_boxes[:, 1] + offsets, widths, heights], axis=1)

    nms_indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), valid_scores.tolist(), conf_threshold, iou_threshold
    )

    detections = []
    if len(nms_indices) > 0:
        for detection_idx in nms_indices.flatten():
            bx1, by1, bx2, by2 = valid_boxes[detection_idx]
            class_id = int(valid_class_ids[detection_idx])
            detections.append({
                'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                'confidence': float(valid_scores[detection_idx]),
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

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        y1_label = max(y1, label_h + 10)
        cv2.rectangle(result_img, (x1, y1_label - label_h - 10), (x1 + label_w, y1_label), color, thickness=cv2.FILLED)
        cv2.putText(result_img, label, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness=1, lineType=cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, result_img)

    return result_img

def main():
    parser = argparse.ArgumentParser(description="Yolov6 Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--nms', type=float, default=0.4)
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
    input_h = int(tensor_attr["dims"][1])
    input_w = int(tensor_attr["dims"][2])
    input_shape = (input_h, input_w)
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    for i, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            input_tensor, original_img, scale, pad = preprocess(image_path, input_shape, s, zp, tensor_type)
            outputs = amlnn.inference(inputs=[input_tensor])
            detections = postprocess(outputs, input_shape, scale, pad, args.conf, args.nms)

            if detections:
                print(f"    Detected {len(detections)} objects:")
                for detection_idx, det in enumerate(detections, 1):
                    print(f"      {detection_idx}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("    No objects detected")

            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            save_path = os.path.join(result_dir, f"{Path(image_path).stem}_result.jpg")
            draw_detections(original_img, detections, save_path)
            print(f"    Result saved to: {save_path}")
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print("=" * 60)
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
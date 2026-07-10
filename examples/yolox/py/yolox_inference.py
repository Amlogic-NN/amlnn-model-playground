# -*- coding: utf-8 -*-

"""
Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import os
import glob
import argparse
import cv2
import colorsys
from pathlib import Path
from amlnn.api import AMLNN

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

CLASS_NAMES = {
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

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=0.017912, zp=-11, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    normalized_img = (rgb_img.astype(np.float32) - MEAN) / STD

    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Only 'NCHW' and 'NHWC' are supported.")

    val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(val, 0, 255).astype(np.uint8)

    return input_tensor, original_img, scale, pad

def postprocess(outputs, scale, pad, class_names, img_size=(640, 640), conf_threshold=0.25, iou_threshold=0.45, p6=False):
    output = outputs[0] if isinstance(outputs, list) else outputs
    output = np.squeeze(output)

    if output.ndim == 2 and output.shape[0] < output.shape[1] and output.shape[0] == 85:
        output = output.T

    obj_conf = output[:, 4]
    cls_scores = output[:, 5:]

    max_cls_scores = np.max(cls_scores, axis=1)
    class_ids = np.argmax(cls_scores, axis=1)
    final_scores = obj_conf * max_cls_scores

    valid_mask = final_scores >= conf_threshold
    if not np.any(valid_mask):
        return []

    valid_preds = output[valid_mask]
    valid_scores = final_scores[valid_mask]
    valid_class_ids = class_ids[valid_mask]

    strides = [8, 16, 32, 64] if p6 else [8, 16, 32]
    grids, expanded_strides = [], []

    for stride in strides:
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0],), stride))

    all_grids = np.concatenate(grids, axis=0)
    all_strides = np.concatenate(expanded_strides, axis=0)

    if all_grids.shape[0] != output.shape[0]:
        raise ValueError(f"Output shape {output.shape[0]} does not match grid anchors {all_grids.shape[0]}")

    valid_grids = all_grids[valid_mask]
    valid_strides = all_strides[valid_mask]

    tx = valid_preds[:, 0]
    ty = valid_preds[:, 1]
    tw = valid_preds[:, 2]
    th = valid_preds[:, 3]

    cx = (tx + valid_grids[:, 0]) * valid_strides
    cy = (ty + valid_grids[:, 1]) * valid_strides
    w = np.exp(tw) * valid_strides
    h = np.exp(th) * valid_strides

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    valid_boxes = np.stack([x1, y1, x2, y2], axis=1)

    pad_x, pad_y = pad
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale
    valid_boxes = np.maximum(valid_boxes, 0)

    detections = []
    unique_classes = np.unique(valid_class_ids)

    for c in unique_classes:
        class_mask = valid_class_ids == c
        c_boxes = valid_boxes[class_mask]
        c_scores = valid_scores[class_mask]

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
                cid = int(c)
                name = class_names[cid] if cid < len(class_names) else f'class_{cid}'
                detections.append({
                    'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
                    'confidence': float(c_scores[idx]),
                    'class_id': cid,
                    'class_name': name
                })

    return detections

def vis(img, detections, conf=0.5, class_names=None):
    result_img = img.copy()
    img_height, img_width = img.shape[:2]
    font_scale = max(0.6, min(1.2, np.sqrt(img_height**2 + img_width**2) * 0.0015))
    thickness = max(2, int(font_scale * 2.5))
    font = cv2.FONT_HERSHEY_SIMPLEX

    def get_color(class_id):
        hue = (class_id * 137.508) % 360
        rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.8, 0.9)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

    for det in detections:
        if det['confidence'] < conf:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        confidence = det['confidence']
        class_id = det['class_id']
        name = det.get('class_name', f'class_{class_id}')

        color = get_color(class_id)
        brightness = sum(color) / 3.0
        txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        text = f'{name}:{confidence * 100:.1f}%'
        txt_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)

        txt_bk_color = [int(c * 0.7) for c in color]
        cv2.rectangle(
            result_img,
            (x1, y1 + 1),
            (x1 + txt_size[0] + 1, y1 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )

        cv2.putText(result_img, text, (x1, y1 + txt_size[1]), font, font_scale, txt_color, thickness=thickness)

    return result_img

def main():
    parser = argparse.ArgumentParser(description="Yolox Demo")
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
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            input_tensor, original_img, scale, pad = preprocess(image_path, new_shape=(640, 640), data_format='NHWC', s=s, zp=zp, tensor_type=tensor_type)

            outputs = amlnn.inference(inputs=[input_tensor])

            detections = postprocess(outputs, scale, pad, CLASS_NAMES, img_size=(640, 640), conf_threshold=0.25, iou_threshold=0.45, p6=False)

            if detections:
                print(f"    Detected {len(detections)} objects:")
                for idx_det, det in enumerate(detections, 1):
                    print(f"      {idx_det}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("    No objects detected")

            img_name = Path(image_path).stem
            save_path = f"{img_name}_result.jpg"
            result_img = vis(original_img, detections, conf=0.25, class_names=CLASS_NAMES)
            cv2.imwrite(save_path, result_img)
            print(f"    Result saved to: {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print(amlnn.get_perf_info())
    amlnn.perf_visualize()

    amlnn.uninit()

if __name__ == "__main__":
    main()
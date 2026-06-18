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

import os
import cv2
import glob
import argparse
import numpy as np
from pathlib import Path
from amlnn.api import AMLNN

MEAN = np.array([0, 0, 0], dtype=np.float32)
STD  = np.array([255, 255, 255], dtype=np.float32)

NAMES = [
    'ok', 'stop', 'palm', 'like', 'dislike', 'no_gesture', 'call', 'fist',
    'four', 'mute', 'one', 'peace', 'peace_inverted', 'rock',
    'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted'
]

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3
INPUT_SIZE = 640

STRIDES = [32.0, 16.0, 8.0]
GRIDS = [20, 40, 80]
ANCHOR_GRIDS = [
    np.array([116, 90, 156, 198, 373, 326], dtype=np.float32).reshape(1, 3, 1, 1, 2),
    np.array([30, 61, 62, 45, 59, 119], dtype=np.float32).reshape(1, 3, 1, 1, 2),
    np.array([10, 13, 16, 30, 33, 23], dtype=np.float32).reshape(1, 3, 1, 1, 2),
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def decode_one_output(y: np.ndarray, g: int, stride: float, anchor_grid: np.ndarray) -> np.ndarray:
    # Format is NHWC: (1, g, g, 72)
    y = y.reshape(1, g, g, 3, 24).transpose(0, 3, 1, 2, 4)

    y = sigmoid(y)

    # Create grid and decode
    yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing='ij')
    grid = np.stack((xv, yv), axis=-1).reshape(1, 1, g, g, 2).astype(np.float32)

    xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
    wh = (y[..., 2:4] * 2.0) ** 2 * anchor_grid
    obj, cls = y[..., 4:5], y[..., 5:]

    decoded = np.concatenate([xy, wh, obj, cls], axis=-1).reshape(1, 3 * g * g, 24)
    return decoded

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=0.003789, zp=-128, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    orig_h, orig_w = original_img.shape[:2]

    resized_img = cv2.resize(original_img, new_shape)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0

    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    raw_val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)

    return input_tensor, original_img, orig_w, orig_h

def postprocess(outputs, orig_w, orig_h, conf_threshold=0.25, iou_threshold=0.45):
    def get_grid_num(x): return int(np.asarray(x).shape[1])
    output_tensors = sorted([np.asarray(o) for o in outputs], key=get_grid_num)

    # Decode
    decoded_all = []
    for i, y in enumerate(output_tensors):
        decoded_all.append(decode_one_output(y, GRIDS[i], STRIDES[i], ANCHOR_GRIDS[i]))
    pred = np.concatenate(decoded_all, axis=1)[0]

    boxes_xywh, obj, cls_scores = pred[:, 0:4], pred[:, 4], pred[:, 5:]
    class_ids = np.argmax(cls_scores, axis=1)
    scores = obj * np.max(cls_scores, axis=1)

    mask = scores > conf_threshold
    if mask.sum() == 0:
        return []

    sel_boxes_xywh = boxes_xywh[mask]
    sel_scores = scores[mask]
    sel_class_ids = class_ids[mask]

    # Convert [center_x, center_y, w, h] to [top_left_x, top_left_y, w, h] for OpenCV
    sel_boxes_tlwh = sel_boxes_xywh.copy()
    sel_boxes_tlwh[:, 0] -= sel_boxes_xywh[:, 2] / 2.0
    sel_boxes_tlwh[:, 1] -= sel_boxes_xywh[:, 3] / 2.0

    final_boxes, final_scores, final_class_ids = [], [], []

    # NMS
    keep = cv2.dnn.NMSBoxes(sel_boxes_tlwh.tolist(), sel_scores.tolist(), conf_threshold, iou_threshold)

    if len(keep) > 0:
        keep = np.array(keep).flatten()
        kept_xywh = sel_boxes_xywh[keep]

        for i, k in enumerate(keep):
            cx, cy, w, h = kept_xywh[i]
            final_boxes.append([cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0])
            final_scores.append(float(sel_scores[k]))
            final_class_ids.append(int(sel_class_ids[k]))

    if not final_boxes:
        return []

    final_boxes = np.asarray(final_boxes, dtype=np.float32)
    
    # Scale boxes back to original image dimensions
    final_boxes[:, [0, 2]] *= (orig_w / float(INPUT_SIZE))
    final_boxes[:, [1, 3]] *= (orig_h / float(INPUT_SIZE))
    final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, orig_w - 1)
    final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, orig_h - 1)

    # Format into standardized list of dictionaries
    order = np.argsort(-np.asarray(final_scores))
    detections = []
    for idx in order:
        bx1, by1, bx2, by2 = final_boxes[idx]
        cid = final_class_ids[idx]
        detections.append({
            'bbox': [float(bx1), float(by1), float(bx2), float(by2)],
            'confidence': float(final_scores[idx]),
            'class_id': int(cid),
            'class_name': NAMES[int(cid)]
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
        confidence = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']
        color = get_class_color(class_id)

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(save_path, result_img)
    return result_img


def main():
    parser = argparse.ArgumentParser(description="Gesture Demo")
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--top1-only', action='store_true', help='Only keep highest score detection')
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
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess input
            input_tensor, original_img, orig_w, orig_h = preprocess(
                image_path,
                new_shape=(INPUT_SIZE, INPUT_SIZE),
                data_format="NHWC",
                s=s,
                zp=zp,
                tensor_type=tensor_type
            )

            # Run inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess results
            detections = postprocess(
                outputs,
                orig_w,
                orig_h,
                conf_threshold=CONF_THRESHOLD,
                iou_threshold=NMS_THRESHOLD
            )

            if args.top1_only and detections:
                detections = [max(detections, key=lambda x: x['confidence'])]

            # Print detection results
            if detections:
                print(f"    Detected {len(detections)} objects:")
                for idx, det in enumerate(detections, 1):
                    print(f"      {idx}. {det['class_name']} ({det['confidence']:.2f})")
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

    # Optional visualization
    amlnn.perf_visualize()

    # Release resources
    amlnn.uninit()

if __name__ == "__main__":
    main()
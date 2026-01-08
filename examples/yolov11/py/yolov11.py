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
from pathlib import Path
from amlnnlite.api import AMLNNLite

class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

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

def preprocess(img_path, new_shape=(640,640)):
    original_img = cv2.imread(str(img_path))
    if original_img is None: return None, None, None, None
    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.transpose(normalized_img, (2,0,1)), 0) # NCHW
    return input_tensor, original_img, scale, pad

def postprocess(outputs, scale, pad, strides=[32,16,8], conf_threshold=0.25, iou_threshold=0.45):
    all_boxes, all_scores, all_class_ids = [], [], []
    for scale_idx, output in enumerate(outputs):
        stride = strides[scale_idx]
        feat = output[0].transpose(1, 2, 0) # H, W, C
        h, w, c = feat.shape
        dfl = feat[:, :, :64].reshape(h, w, 4, 16)
        cls_logits = feat[:, :, 64:]
        cls_scores = 1.0 / (1.0 + np.exp(-cls_logits)) # sigmoid
        
        exp_x = np.exp(dfl - np.max(dfl, axis=-1, keepdims=True))
        p = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        bbox_deltas = np.sum(p * np.arange(16, dtype=np.float32), axis=-1)
        
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        l, t, r, b = np.split(bbox_deltas, 4, axis=-1)
        x1, y1 = (grid_x + 0.5 - l[..., 0]) * stride, (grid_y + 0.5 - t[..., 0]) * stride
        x2, y2 = (grid_x + 0.5 + r[..., 0]) * stride, (grid_y + 0.5 + b[..., 0]) * stride
        
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4))
        all_scores.append(cls_scores.reshape(-1, cls_scores.shape[-1]))
    
    final_boxes = np.concatenate(all_boxes, axis=0)
    final_scores_all = np.concatenate(all_scores, axis=0)
    final_class_ids = np.argmax(final_scores_all, axis=1)
    final_scores = np.max(final_scores_all, axis=1)

    mask = final_scores > conf_threshold
    if not np.any(mask): return []
    
    valid_boxes = final_boxes[mask]
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad[0]) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad[1]) / scale
    
    indices = cv2.dnn.NMSBoxes(valid_boxes.tolist(), final_scores[mask].tolist(), conf_threshold, iou_threshold)
    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detections.append({
                'bbox': valid_boxes[idx].tolist(),
                'confidence': float(final_scores[mask][idx]),
                'class_name': class_names.get(int(final_class_ids[mask][idx]), 'unknown')
            })
    return detections

def main():
    parser = argparse.ArgumentParser(description="YOLOV11 AMLNNLite Demo")
    parser.add_argument('--board-work-path', default='/data/nn', help='Work path on board')
    parser.add_argument('-m', '--model-path', required=True, help='Path to .adla or .tflite model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--run-cycles', type=int, default=1, help='Inference cycles for profiling')
    parser.add_argument('--loglevel', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    args = parser.parse_args()

    amlnn = AMLNNLite()
    amlnn.config(
        board_work_path=args.board_work_path,
        model_path=args.model_path,
        run_cycles=args.run_cycles,
        loglevel=args.loglevel
    )
    amlnn.init()

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {args.image_dir}")
        amlnn.uninit(); return

    model_stem = Path(args.model_path).stem
    res_dir = f"{model_stem}_result"
    os.makedirs(res_dir, exist_ok=True)

    for i, img_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(img_path)}")
        print("=" * 60)
        
        input_tensor, ori_img, scale, pad = preprocess(img_path)
        if input_tensor is None: continue
        
        for _ in range(args.run_cycles):
            outputs = amlnn.inference(input_tensor, inputs_data_format='NCHW', outputs_data_format='NCHW')
        
        detections = postprocess(outputs, scale, pad)
        
        print(f"    Detected {len(detections)} objects:")
        for idx, det in enumerate(detections, 1):
            print(f"      {idx}. {det['class_name']} ({det['confidence']:.2f})")
            
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(ori_img, f"{det['class_name']} {det['confidence']:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        save_path = os.path.join(res_dir, f"{Path(img_path).stem}_result.jpg")
        cv2.imwrite(save_path, ori_img)
        print(f"    Result saved to: {save_path}")

    amlnn.visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
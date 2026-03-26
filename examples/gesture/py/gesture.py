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
from amlnnlite.api import AMLNNLite


NAMES = [
    'ok', 'stop', 'palm', 'like', 'dislike', 'no_gesture', 'call', 'fist',
    'four', 'mute', 'one', 'peace', 'peace_inverted', 'rock',
    'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted'
]

INPUT_SIZE = 640

STRIDES = [32.0, 16.0, 8.0]
GRIDS = [20, 40, 80]
ANCHOR_GRIDS = [
    np.array([116, 90, 156, 198, 373, 326], dtype=np.float32).reshape(1, 3, 1, 1, 2),
    np.array([30, 61, 62, 45, 59, 119], dtype=np.float32).reshape(1, 3, 1, 1, 2),
    np.array([10, 13, 16, 30, 33, 23], dtype=np.float32).reshape(1, 3, 1, 1, 2),
]


def preprocess_bgr(bgr: np.ndarray):
    h0, w0 = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    rgb = rgb.astype(np.float32) / 255.0

    nchw = np.transpose(rgb, (2, 0, 1))[None, ...]  
    nhwc = np.transpose(nchw, (0, 2, 3, 1))       

    return nhwc, w0, h0


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45):
    order = np.argsort(-scores)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = box_iou_one(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]

    return keep


def decode_one_output(y: np.ndarray, g: int, stride: float, anchor_grid: np.ndarray) -> np.ndarray:
    y = np.transpose(y, (0, 1, 3, 2))      
    y = y.reshape(1, g, g, 3, 24)          
    y = np.transpose(y, (0, 3, 1, 2, 4))    

    yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing='ij')
    grid = np.stack((xv, yv), axis=-1).reshape(1, 1, g, g, 2).astype(np.float32)

    xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
    wh = (y[..., 2:4] * 2.0) ** 2 * anchor_grid
    obj = y[..., 4:5]
    cls = y[..., 5:]

    decoded = np.concatenate([xy, wh, obj, cls], axis=-1)  
    decoded = decoded.reshape(1, 3 * g * g, 24)            
    return decoded


def decode_outputs(output_tensors):
    decoded_all = []
    for i, y in enumerate(output_tensors):
        decoded = decode_one_output(
            y=y,
            g=GRIDS[i],
            stride=STRIDES[i],
            anchor_grid=ANCHOR_GRIDS[i]
        )
        decoded_all.append(decoded)

    pred = np.concatenate(decoded_all, axis=1)  
    return pred[0]                              


def postprocess(pred: np.ndarray, conf_thres: float = 0.25, nms_thres: float = 0.45):
    boxes_xywh = pred[:, 0:4]
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]   

    class_ids = np.argmax(cls_scores, axis=1)
    class_scores = np.max(cls_scores, axis=1)
    scores = obj * class_scores

    mask = scores > conf_thres
    if mask.sum() == 0:
        return [], [], []

    sel_boxes_xywh = boxes_xywh[mask]
    sel_scores = scores[mask]
    sel_class_ids = class_ids[mask]

    sel_boxes_xyxy = xywh2xyxy(sel_boxes_xywh)

    final_boxes = []
    final_scores = []
    final_class_ids = []

    unique_classes = np.unique(sel_class_ids)
    for cid in unique_classes:
        cls_mask = sel_class_ids == cid
        cls_boxes = sel_boxes_xyxy[cls_mask]
        cls_scores_part = sel_scores[cls_mask]

        keep = nms(cls_boxes, cls_scores_part, iou_thres=nms_thres)
        for k in keep:
            final_boxes.append(cls_boxes[k].copy())
            final_scores.append(float(cls_scores_part[k]))
            final_class_ids.append(int(cid))

    if len(final_boxes) == 0:
        return [], [], []

    final_boxes = np.asarray(final_boxes, dtype=np.float32)
    final_scores = np.asarray(final_scores, dtype=np.float32)
    final_class_ids = np.asarray(final_class_ids, dtype=np.int32)

    order = np.argsort(-final_scores)
    return final_boxes[order], final_scores[order], final_class_ids[order]


def scale_boxes_to_original(boxes_xyxy: np.ndarray, orig_w: int, orig_h: int):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy

    scale_x = orig_w / float(INPUT_SIZE)
    scale_y = orig_h / float(INPUT_SIZE)

    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)

    return boxes


def draw_detections(bgr: np.ndarray, boxes, scores, class_ids):
    vis = bgr.copy()
    h, w = vis.shape[:2]

    font_scale = max(0.8, min(w, h) / 600.0)
    font_thickness = max(2, int(min(w, h) / 300))
    box_thickness = max(2, int(min(w, h) / 250))

    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = f'{NAMES[int(cid)]} {float(score):.2f}'

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        text_y = max(30, y1 - 10)
        cv2.putText(
            vis, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA
        )

    return vis


def infer_bgr(amlnn, bgr, conf_thresh=0.25, nms_thresh=0.45):
    inp, orig_w, orig_h = preprocess_bgr(bgr)

    outputs = amlnn.inference(inp, inputs_data_format='NHWC')
    output_tensors = [np.asarray(out) for out in outputs]

    def get_grid_num(x):
        s = np.asarray(x).shape
        if len(s) != 4:
            raise ValueError(f"Unexpected output shape: {s}")
        return int(s[1])

    output_tensors = sorted(output_tensors, key=get_grid_num)  # 400, 1600, 6400

    pred = decode_outputs(output_tensors)
    boxes, scores, class_ids = postprocess(pred, conf_thres=conf_thresh, nms_thres=nms_thresh)
    boxes = scale_boxes_to_original(boxes, orig_w, orig_h)

    boxes_xyxy = [tuple(map(int, box)) for box in boxes]
    scores = [float(x) for x in scores]
    class_ids = [int(x) for x in class_ids]

    return boxes_xyxy, scores, class_ids


def main():
    parser = argparse.ArgumentParser(description="Gesture AMLNNLite Demo")
    parser.add_argument('--board-work-path', type=str, default='/data/local/tmp')
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory of test images')
    parser.add_argument('--run-cycles', type=int, default=1, help='Inference cycles')
    parser.add_argument('--loglevel', type=str, default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--nms-thres', type=float, default=0.3)
    parser.add_argument('--top1-only', action='store_true', help='Only keep the highest score detection')
    args = parser.parse_args()

    amlnn = AMLNNLite()
    amlnn.config(
        board_work_path=args.board_work_path,
        model_path=args.model_path,
        run_cycles=args.run_cycles,
        loglevel=args.loglevel
    )
    amlnn.init()

    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.[jp][pn][g]")))
    if not image_files:
        print(f"No images found in {args.image_dir}")
        amlnn.uninit()
        return

    res_dir = "gesture_result"
    os.makedirs(res_dir, exist_ok=True)

    for idx, img_path in enumerate(image_files, start=1):
        print("=" * 60)
        print(f"Processing image {idx}/{len(image_files)}: {Path(img_path).name}")
        print("=" * 60)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        boxes, scores, class_ids = infer_bgr(
            amlnn, img,
            conf_thresh=args.conf_thres,
            nms_thresh=args.nms_thres
        )

        if args.top1_only and len(boxes) > 0:
            max_idx = int(np.argmax(np.array(scores)))
            boxes = [boxes[max_idx]]
            scores = [scores[max_idx]]
            class_ids = [class_ids[max_idx]]

        if len(boxes) == 0:
            print("    No objects detected")
            vis = img.copy()
        else:
            print(f"    Detected {len(boxes)} objects:")
            for i, (box, score, cid) in enumerate(zip(boxes, scores, class_ids), 1):
                print(f"      {i}. class={NAMES[int(cid)]}")
                print(f"         score={float(score):.3f}")
                print(f"         box={list(map(int, box))}")
            vis = draw_detections(img, np.array(boxes), np.array(scores), np.array(class_ids))

        save_path = os.path.join(res_dir, Path(img_path).name)
        cv2.imwrite(save_path, vis)
        print(f"    Result saved to: {save_path}")

    if args.loglevel == 'INFO':
        print("\nPerformance analysis visualization starting...")

    amlnn.visualize()
    amlnn.uninit()


if __name__ == "__main__":
    main()
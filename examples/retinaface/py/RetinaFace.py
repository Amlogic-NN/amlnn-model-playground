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
import time
import numpy as np
from pathlib import Path
from amlnnlite.api import AMLNNLite

class PriorBox:
    def __init__(self, image_size=(320, 320)):
        self.image_size = image_size
        self.steps = [8, 16, 32]
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]

    def forward(self):
        priors = []
        h, w = self.image_size
        for idx, step in enumerate(self.steps):
            fm_h, fm_w = int(np.ceil(h / step)), int(np.ceil(w / step))
            for i in range(fm_h):
                for j in range(fm_w):
                    for min_size in self.min_sizes[idx]:
                        cx, cy = (j + 0.5) * step / w, (i + 0.5) * step / h
                        s_kx, s_ky = min_size / w, min_size / h
                        priors.append([cx, cy, s_kx, s_ky])
        return np.array(priors, dtype=np.float32)

def decode_boxes(loc, priors, variances=(0.1, 0.2)):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landmarks(pre, priors, variances=(0.1, 0.2)):
    landms = np.concatenate([
        priors[:, :2] + pre[:, i:i+2] * variances[0] * priors[:, 2:] for i in range(0, 10, 2)
    ], axis=1)
    return landms

def nms(dets, thresh=0.4):
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.maximum(y1[i], y1[order[1:]]), np.maximum(y1[i], y1[order[1:]]) # fix
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - (w * h))
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep

def postprocess_retinaface(outputs, priors, conf_thresh=0.5, nms_thresh=0.4):
    loc = conf = landms = None
    for out in outputs:
        out = np.squeeze(np.asarray(out))
        if out.shape[-1] == 4: loc = out
        elif out.shape[-1] == 2: conf = out
        elif out.shape[-1] == 10: landms = out

    if loc is None or conf is None or landms is None: return [], [], []
    scores = conf[:, 1]
    mask = scores > conf_thresh
    if not np.any(mask): return [], [], []
    
    boxes = decode_boxes(loc[mask], priors[mask])
    landms = decode_landmarks(landms[mask], priors[mask])
    scores = scores[mask]
    keep = nms(np.hstack((boxes, scores[:, None])), nms_thresh)
    return boxes[keep], landms[keep], scores[keep]

def preprocess(img_path, input_size=(320, 320)):
    img = cv2.imread(img_path)
    if img is None: return None, None, 0, 0, 0
    h0, w0 = img.shape[:2]
    scale = min(input_size[0] / w0, input_size[1] / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((input_size[1], input_size[0], 3), 128, dtype=np.uint8)
    pad_x, pad_y = (input_size[0] - nw) // 2, (input_size[1] - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return np.expand_dims(canvas.astype(np.float32), axis=0), img, scale, pad_x, pad_y

def main():
    parser = argparse.ArgumentParser(description="RetinaFace AMLNNLite Demo")
    parser.add_argument('--board-work-path', type=str, default='/data/nn')
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory of test images')
    parser.add_argument('--run-cycles', type=int, default=1, help='Inference cycles')
    parser.add_argument('--loglevel', type=str, default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    amlnn = AMLNNLite()
    amlnn.config(board_work_path=args.board_work_path, 
                 model_path=args.model_path, 
                 run_cycles=args.run_cycles, 
                 loglevel=args.loglevel)
    amlnn.init()

    priors = PriorBox((320, 320)).forward()
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.[jp][pn][g]")))
    
    if not image_files:
        print(f"No images found in {args.image_dir}")
        amlnn.uninit(); return

    res_dir = "retinaface_result"
    os.makedirs(res_dir, exist_ok=True)

    for idx, img_path in enumerate(image_files, start=1):
        print("=" * 60)
        print(f"Processing image {idx}/{len(image_files)}: {Path(img_path).name}")
        print("=" * 60)

        inp, orig, scale, pad_x, pad_y = preprocess(img_path)
        if inp is None: continue
        
        outputs = amlnn.inference(inp, inputs_data_format='NHWC')

        boxes, landms, scores = postprocess_retinaface(outputs, priors)

        if len(boxes) > 0:
            print(f"    Detected {len(boxes)} objects:")
            for i, sc in enumerate(scores, 1):
                print(f"      {i}. face ({sc:.2f})")
        else:
            print("    No objects detected")

        for box, lm in zip(boxes, landms):
            x1 = int((box[0] * 320 - pad_x) / scale)
            y1 = int((box[1] * 320 - pad_y) / scale)
            x2 = int((box[2] * 320 - pad_x) / scale)
            y2 = int((box[3] * 320 - pad_y) / scale)
            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for lx, ly in lm.reshape(5, 2):
                cv2.circle(orig, (int((lx*320-pad_x)/scale), int((ly*320-pad_y)/scale)), 2, (0, 0, 255), -1)

        save_path = os.path.join(res_dir, Path(img_path).name)
        cv2.imwrite(save_path, orig)
        print(f"    Result saved to: {save_path}")

    if args.loglevel == 'INFO':
        print("\nI Performance analysis visualization starting...")
        
    amlnn.visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
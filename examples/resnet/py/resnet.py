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
import argparse
import glob
import cv2
import numpy as np
from pathlib import Path
from amlnnlite.api import AMLNNLite

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 58.395, 58.395], dtype=np.float32)

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)

    img = (img - MEAN) / STD
    img = np.expand_dims(img, axis=0)  
    return img

def postprocess_topk(logits, labels, k=5):
    logits = logits.squeeze()
    idx = np.argsort(logits)[::-1][:k]

    print(f"\n    Top-{k} Results:")
    for i, c in enumerate(idx):
        name = labels[c] if c < len(labels) else f"Unknown({c})"
        score = logits[c]
        print(f"      {i+1}. {name:20s}  score={score:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Classification AMLNNLite Demo")

    parser.add_argument('--board-work-path', default='/data/nn', help='Work path on board')
    parser.add_argument('--model-path', required=True, help='Path to .adla or .tflite model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--labels', required=True, help='Path to synset_words.txt or labels.txt')
    parser.add_argument('--run-cycles', type=int, default=1, help='Number of inference cycles')
    parser.add_argument('--loglevel', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    amlnn = AMLNNLite()
    amlnn.config(
        board_work_path=args.board_work_path,
        model_path=args.model_path,
        run_cycles=args.run_cycles,
        loglevel=args.loglevel
    )
    amlnn.init()

    if not os.path.exists(args.labels):
        print(f"Error: Label file not found: {args.labels}")
        amlnn.uninit(); return
        
    with open(args.labels, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files.sort()

    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        amlnn.uninit(); return

    total_images = len(image_files)

    for idx, img_path in enumerate(image_files, start=1):
        print("=" * 60)
        print(f"Processing image {idx}/{total_images}: {os.path.basename(img_path)}")
        print("=" * 60)

        inp = preprocess(img_path)
        if inp is None:
            print(f"    Skip: Cannot read {img_path}")
            continue

        for _ in range(args.run_cycles):
            outputs = amlnn.inference(
                inp,
                inputs_data_format='NHWC',
                outputs_data_format='NHWC'
            )

        postprocess_topk(outputs[0], labels, k=5)

    amlnn.visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
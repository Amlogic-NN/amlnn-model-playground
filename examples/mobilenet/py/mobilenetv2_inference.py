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
from PIL import Image
from amlnn.api import AMLNN

MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)
STD  = np.array([58.395, 57.120, 57.375], dtype=np.float32)
TOPK = 5

def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: name for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not load class names from '{path}'. Fallback to generic IDs.")
        return {}

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=0.003789, zp=-128, tensor_type=2):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"can't read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)

    # NOTE: Change this based on which model you are using
    img = (img / 127.5) - 1.0 # TFLite Normalization
    # img = (img - MEAN) / STD # Onnx ImageNet Normalization

    if data_format == 'NCHW':
        input_tensor = np.transpose(img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    raw_val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)

    return input_tensor


def postprocess_topk(logits, labels, k=5):
    logits = logits.squeeze()
    idx = np.argsort(logits)[::-1][:k]

    print(f"\n    Top-{k} Results:")
    for i, c in enumerate(idx):
        name = labels[c] if c < len(labels) else f"Unknown({c})"
        score = logits[c]
        print(f"      {i+1}. {name:20s}  score={score:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Mobilenet Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--labels', required=True, help='Path to labels.txt')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    if not os.path.exists(args.labels):
        print(f"Error: Label file not found: {args.labels}")
        amlnn.uninit(); return

    # load labels
    labels = load_class_names(args.labels)

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
            input_tensor = preprocess(image_path, new_shape = (224, 224), data_format ="NHWC", s=s, zp=zp, tensor_type=tensor_type)

            # Run inference
            outputs = amlnn.inference(
                inputs=[input_tensor]
            )

            postprocess_topk(outputs[0], labels, k=TOPK)

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()
    print(f"=" * 60)
    print(amlnn.get_perf_info())

    # Optional visualization
    amlnn.perf_visualize()
    # Release resources
    amlnn.uninit

if __name__ == "__main__":
    main()

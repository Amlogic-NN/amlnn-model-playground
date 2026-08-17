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

def get_quant_dtype(tensor_type):
    if tensor_type == 2:
        return np.int8
    if tensor_type == 3:
        return np.uint8
    raise ValueError(f"Unsupported quantized tensor type: {tensor_type}")

def load_image(img_path, image_shape):
    extension = os.path.splitext(img_path)[1].lower()

    if extension in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"can't read image: {img_path}")
        return img

    if extension == ".txt":
        with open(img_path, "r") as f:
            values = np.fromstring(f.read(), sep=" ", dtype=np.int32)

        height, width = image_shape
        expected_size = height * width * 3
        if values.size != expected_size:
            raise ValueError(
                f"invalid .txt image data size: expected {expected_size} values for "
                f"{height}x{width}x3, got {values.size}"
            )

        if np.any((values < 0) | (values > 255)):
            raise ValueError(f".txt image pixel values must be in [0, 255]: {img_path}")

        return values.astype(np.uint8).reshape(height, width, 3)

    raise ValueError(f"unsupported image input file: {img_path}")

def load_quantized_input(input_path, input_shape, tensor_type):
    extension = os.path.splitext(input_path)[1].lower()
    dtype = get_quant_dtype(tensor_type)
    model_input_shape = (1, input_shape[0], input_shape[1], 3)
    expected_size = int(np.prod(model_input_shape))

    if extension == ".bin":
        values = np.fromfile(input_path, dtype=dtype)
    elif extension == ".qtxt":
        with open(input_path, "r") as f:
            values = np.fromstring(f.read(), sep=" ", dtype=np.int32)

        if tensor_type == 2 and np.any((values < -128) | (values > 127)):
            raise ValueError(f".qtxt int8 values must be in [-128, 127]: {input_path}")
        if tensor_type == 3 and np.any((values < 0) | (values > 255)):
            raise ValueError(f".qtxt uint8 values must be in [0, 255]: {input_path}")

        values = values.astype(dtype)
    else:
        raise ValueError(f"unsupported quantized input file: {input_path}")

    if values.size != expected_size:
        raise ValueError(
            f"invalid {extension} input size: expected {expected_size} values for "
            f"model input {model_input_shape}, got {values.size}"
        )

    return values.reshape(model_input_shape)

def preprocess(img_path, new_shape, s, zp, tensor_type):
    img = load_image(img_path, new_shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)

    # NOTE: Change this based on which model you are using
    # img = (img / 127.5) - 1.0 # TFLite Normalization
    img = (img - MEAN) / STD # Onnx ImageNet Normalization

    input_tensor = np.expand_dims(img, axis=0)

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
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--labels', required=True, help='Path to labels.txt')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.adla)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    if not os.path.exists(args.labels):
        print(f"Error: Label file not found: {args.labels}")
        amlnn.uninit(); return

    # load labels
    labels = load_class_names(args.labels)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.txt", "*.bin", "*.qtxt"]
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

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            extension = os.path.splitext(image_path)[1].lower()

            # Preprocess image inputs; .bin and .qtxt are already quantized model inputs.
            if extension in (".bin", ".qtxt"):
                input_tensor = load_quantized_input(image_path, input_shape, tensor_type)
            else:
                input_tensor = preprocess(image_path, input_shape, s, zp, tensor_type)

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
    # amlnn.perf_visualize()

    # Release resources
    amlnn.uninit

if __name__ == "__main__":
    main()
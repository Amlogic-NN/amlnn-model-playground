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
from amlnn.api import AMLNN

MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)
STD  = np.array([58.395, 57.120, 57.375], dtype=np.float32)

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"can't read image: {img_path}")
    return img

def preprocess(img, new_shape, s, zp, tensor_type):
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    else:
        raise ValueError(f"Unsupported quantized tensor type: {tensor_type}")

    return input_tensor

def main():
    parser = argparse.ArgumentParser(description="Prepare MobileNet .txt, .bin and .qtxt test inputs")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing source images')
    parser.add_argument('--output-dir', default='generated_inputs', help='Directory for generated .txt, .bin and .qtxt files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=False)
    amlnn.load_model(path=args.adla)

    tensor_info = amlnn.get_tensor_info()
    tensor_attr = tensor_info["inputs"][0]

    input_h = int(tensor_attr["dims"][1])
    input_w = int(tensor_attr["dims"][2])
    input_shape = (input_h, input_w)
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    print(amlnn.get_sdk_version())
    print(f"Input dimensions: {tensor_attr['dims']}")
    print(f"Input shape: {input_shape}")
    print(f"Scale: {s}")
    print(f"Zero point: {zp}")
    print(f"Tensor type: {tensor_type}")
    print()

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No source image files found in {args.image_dir}")
        amlnn.uninit()
        return 0

    print(f"Found {len(image_files)} source image file(s):")
    for image_path in image_files:
        print(f"  - {os.path.basename(image_path)}")
    print()

    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            img = load_image(image_path)
            input_tensor = preprocess(img, input_shape, s, zp, tensor_type)

            stem = os.path.splitext(os.path.basename(image_path))[0]
            txt_path = os.path.join(args.output_dir, f"{stem}.txt")
            bin_path = os.path.join(args.output_dir, f"{stem}.bin")
            qtxt_path = os.path.join(args.output_dir, f"{stem}.qtxt")

            np.savetxt(txt_path, img.reshape(-1), fmt="%d")
            input_tensor.tofile(bin_path)
            np.savetxt(qtxt_path, input_tensor.reshape(-1), fmt="%d")

            print(f"  .txt:  {txt_path}")
            print(f"  .bin:  {bin_path}")
            print(f"  .qtxt: {qtxt_path}")
            print(f"  TXT source shape: {img.shape}")
            print(f"  Tensor shape: {input_tensor.shape}")
            print(f"  Tensor dtype: {input_tensor.dtype}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()
    print(f"=" * 60)
    amlnn.uninit()

if __name__ == "__main__":
    main()
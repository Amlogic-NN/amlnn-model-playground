"""
Copyright (C) 2026 Amlogic, Inc. All rights reserved.

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
from amlnn.api import AMLNN

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Cityscapes 19-class color map (in BGR format for OpenCV)
CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102],
    [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220],
    [35, 142, 107], [152, 251, 152], [180, 130, 70], [60, 20, 220],
    [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0],
    [100, 80, 0], [230, 0, 0], [32, 11, 119]
], dtype=np.uint8)

def preprocess(img_path, new_shape, s, zp, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    # Direct resize: new_shape is (height, width), while OpenCV expects (width, height).
    resized_img = cv2.resize(
        original_img, (new_shape[1], new_shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    normalized_img = (rgb_img.astype(np.float32) - MEAN) / STD

    # Quantize the already-normalized values using the input tensor metadata.
    if tensor_type == 0:
        input_tensor = normalized_img.astype(np.float32)
    else:
        raw_value = np.round(normalized_img / s + zp)

        if tensor_type == 2:
            input_tensor = np.clip(raw_value, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_value, 0, 255).astype(np.uint8)
        elif tensor_type == 4:
            input_tensor = np.clip(raw_value, -32768, 32767).astype(np.int16)
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, original_img

def postprocess(outputs, original_shape):
    logits = outputs[0]
    logits = np.squeeze(logits)

    if logits.shape[0] == 19:
        pred_mask = np.argmax(logits, axis=0)
    else:
        pred_mask = np.argmax(logits, axis=-1)

    pred_mask = pred_mask.astype(np.uint8)

    orig_h, orig_w = original_shape[:2]
    pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return pred_mask_resized

def draw_segmentation(img, pred_mask, save_path, alpha):
    print(f"    Applying color map and blending...")
    color_mask = CITYSCAPES_COLORS[pred_mask]

    blended = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)

    cv2.imwrite(save_path, blended)
    print(f"    Image saved to: {save_path}")

    return blended

def main():
    parser = argparse.ArgumentParser(description="PP-LiteSeg Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--mask-alpha', type=float, default=0.5)
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()
    print(f"SDK Version: {amlnn.get_sdk_version()}")

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
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # 1. Preprocess
            input_tensor, original_img = preprocess(
                image_path, input_shape, s, zp, tensor_type
            )

            # 2. Inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # 3. Postprocess
            pred_mask = postprocess(outputs, original_img.shape)

            # 4. Save Paths
            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            # 5. Draw and Save
            draw_segmentation(original_img, pred_mask, str(save_path), args.mask_alpha)

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print(f"=" * 60)
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
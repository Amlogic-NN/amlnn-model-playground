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

MEAN = np.array([122.7709383, 116.7460125, 104.09373615], dtype=np.float32)
STD = np.array([68.5005327, 66.6321579, 70.32316305], dtype=np.float32)
TOP_K = 5

def resize_and_center_crop(img, new_shape):
    shape = img.shape[:2]  # [height, width]

    scale = max(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_size = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    top = (new_size[1] - new_shape[0]) // 2
    left = (new_size[0] - new_shape[1]) // 2
    return resized_img[top:top + new_shape[0], left:left + new_shape[1]]

def preprocess(img_path, new_shape, s, zp, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    # 1. Resize shorter side and center crop
    processed_img = resize_and_center_crop(original_img, new_shape)

    # 2. BGR to RGB
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    rgb_float = rgb_img.astype(np.float32)

    # 3. EVA-02 CLIP normalization and quantization if needed
    normalized = (rgb_float - MEAN) / STD

    if tensor_type == 0:  # FP32 & FP16
        input_tensor = normalized
    elif tensor_type in (2, 3, 4):
        raw_val = np.round(normalized / s + zp)

        if tensor_type == 2:    # Int8
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:  # Uint8
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:                   # Int16
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, original_img

def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: name for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not load class names from '{path}'. Fallback to generic IDs.")
        return {}

def postprocess(outputs, class_names, top_k=TOP_K):
    logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)

    top_indices = np.argsort(probabilities)[::-1][:min(top_k, probabilities.size)]

    results = []
    for class_id in top_indices:
        results.append({
            "class_id": int(class_id),
            "class_name": class_names.get(int(class_id), f"class_{int(class_id)}"),
            "confidence": float(probabilities[int(class_id)])
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="EVA-02 ADLA Image Classification Demo")
    parser.add_argument("--adla", required=True, help="Path to .adla model")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--labels", required=True, help="Path to class labels .txt file")
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.adla)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        amlnn.uninit()
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

    class_names = load_class_names(args.labels)

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess input
            input_tensor, _ = preprocess(image_path, input_shape, s, zp, tensor_type)

            # Run inference
            outputs = amlnn.inference(
                inputs=[input_tensor]
            )

            # Postprocess results
            results = postprocess(outputs, class_names)

            # Print classification results
            print(f"    Top {len(results)} results:")
            for rank, result in enumerate(results, 1):
                print(f"      {rank}. {result['class_name']} ({result['confidence']:.4f})")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()
    print(f"=" * 60)
    print(amlnn.get_perf_info())

    # Optional visualization
    # amlnn.perf_visualize()

    # Release resources
    amlnn.uninit()

if __name__ == "__main__":
    main()
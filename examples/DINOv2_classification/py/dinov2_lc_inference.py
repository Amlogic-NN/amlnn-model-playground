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

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return {idx: name for idx, name in enumerate(names)}
    except Exception as e:
        print(f"Warning: Could not load class names from '{path}'. Fallback to generic IDs.")
        return {}


def prepare_input_tensor(input_float, s, zp, tensor_type):
    if tensor_type == 0:
        return input_float.astype(np.float32)

    raw_val = np.round(input_float / s + zp)

    if tensor_type == 2:
        return np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        return np.clip(raw_val, 0, 255).astype(np.uint8)
    elif tensor_type == 4:
        return np.clip(raw_val, -32768, 32767).astype(np.int16)

    raise ValueError(f"Does not support tensor type: {tensor_type}")


def preprocess(image_path, new_shape, s, zp, tensor_type):
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        raise ValueError(f"can't read image: {image_path}")

    input_height, input_width = new_shape
    height, width = original_img.shape[:2]

    if height < width:
        resized_height = 256
        resized_width = round(width * 256 / height)
    else:
        resized_width = 256
        resized_height = round(height * 256 / width)

    resized_img = cv2.resize(original_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

    crop_x = (resized_width - input_width) // 2
    crop_y = (resized_height - input_height) // 2
    cropped_img = resized_img[crop_y:crop_y + input_height, crop_x:crop_x + input_width]

    rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    input_float = rgb_img.astype(np.float32)
    input_float = (input_float - IMAGENET_MEAN) / IMAGENET_STD
    input_float = np.expand_dims(input_float, axis=0)

    return prepare_input_tensor(input_float, s, zp, tensor_type)


def softmax(logits):
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities)


def postprocess(output, class_names, topk):
    logits = np.asarray(output, dtype=np.float32).reshape(-1)
    probabilities = softmax(logits)
    top_indices = np.argsort(probabilities)[-topk:][::-1]

    results = []
    for class_id in top_indices:
        class_id = int(class_id)
        results.append({
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            "confidence": float(probabilities[class_id])
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Linear Classification Single-Input ADLA Demo")
    parser.add_argument("--backbone-model", required=True, help="Path to DINOv2 ViT-S/14 backbone .adla model")
    parser.add_argument("--classifier-model", required=True, help="Path to DINOv2 single-input linear classifier .adla model")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--labels", required=True, help="Path to ImageNet class names")
    parser.add_argument("--topk", type=int, default=5, help="Number of results to print")
    args = parser.parse_args()

    print("Running inference script:", os.path.abspath(__file__))
    print("Classifier interface: one concatenated input")
    print()

    class_names = load_class_names(args.labels)

    backbone = AMLNN()
    backbone.init_runtime(mode="native", enable_perf=True)
    backbone.load_model(path=args.backbone_model)
    backbone_info = backbone.get_tensor_info()

    classifier = AMLNN()
    classifier.init_runtime(mode="native", enable_perf=True)
    classifier.load_model(path=args.classifier_model)
    classifier_info = classifier.get_tensor_info()

    print(backbone.get_sdk_version())

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        backbone.uninit()
        classifier.uninit()
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")
    for image_file in image_files:
        print(f"  - {os.path.basename(image_file)}")
    print()

    if len(backbone_info["outputs"]) != 4:
        raise RuntimeError(f"Expected 4 backbone outputs, got {len(backbone_info['outputs'])}")

    if len(classifier_info["inputs"]) != 1:
        raise RuntimeError(f"Expected 1 classifier input, got {len(classifier_info['inputs'])}")

    backbone_input_attr = backbone_info["inputs"][0]
    backbone_input_h = int(backbone_input_attr["dims"][1])
    backbone_input_w = int(backbone_input_attr["dims"][2])
    backbone_input_shape = (backbone_input_h, backbone_input_w)
    backbone_input_s = float(backbone_input_attr["scale"])
    backbone_input_zp = int(backbone_input_attr["zp"])
    backbone_input_type = int(backbone_input_attr["type"])

    classifier_input_attr = classifier_info["inputs"][0]
    classifier_input_s = float(classifier_input_attr["scale"])
    classifier_input_zp = int(classifier_input_attr["zp"])
    classifier_input_type = int(classifier_input_attr["type"])

    print(f"Backbone input: name={backbone_input_attr['name']}, shape={backbone_input_attr['dims']}")
    for i, output in enumerate(backbone_info["outputs"]):
        print(f"Backbone output {i}: name={output['name']}, shape={output['dims']}")
    print(f"Classifier input: name={classifier_input_attr['name']}, shape={classifier_input_attr['dims']}")
    print(f"Classifier output: name={classifier_info['outputs'][0]['name']}, shape={classifier_info['outputs'][0]['dims']}")
    print()

    for i, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            backbone_input = preprocess(
                image_path,
                backbone_input_shape,
                backbone_input_s,
                backbone_input_zp,
                backbone_input_type
            )

            backbone_outputs = backbone.inference(inputs=[backbone_input])

            concat_features = np.concatenate([
                np.asarray(backbone_outputs[0], dtype=np.float32),
                np.asarray(backbone_outputs[1], dtype=np.float32),
                np.asarray(backbone_outputs[2], dtype=np.float32),
                np.asarray(backbone_outputs[3], dtype=np.float32)
            ], axis=-1).astype(np.float32)

            classifier_input = prepare_input_tensor(
                concat_features,
                classifier_input_s,
                classifier_input_zp,
                classifier_input_type
            )

            classifier_outputs = classifier.inference(inputs=[classifier_input])
            results = postprocess(classifier_outputs[0], class_names, args.topk)

            print("    Results:")
            for rank, result in enumerate(results, 1):
                print(f"      {rank}. {result['class_name']} ({result['confidence']:.4f})")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print("=" * 60)
    print("Backbone performance:")
    print(backbone.get_perf_info())
    print("Classifier performance:")
    print(classifier.get_perf_info())

    backbone.uninit()
    classifier.uninit()


if __name__ == "__main__":
    main()
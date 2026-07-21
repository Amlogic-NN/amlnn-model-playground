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
from transformers import CLIPTokenizer
from amlnn.api import AMLNN

MEAN = np.array([122.7709383, 116.7460125, 104.09373615], dtype=np.float32)
STD = np.array([68.5005327, 66.6321579, 70.32316305], dtype=np.float32)
MAX_TEXT_LENGTH = 64


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def l2_normalize(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def preprocess_image(img_path, input_shape, scale, zero_point, tensor_type):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    input_h, input_w = input_shape
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb_img.shape[:2]
    resize_scale = max(input_w / original_w, input_h / original_h)
    resized_w = int(round(original_w * resize_scale))
    resized_h = int(round(original_h * resize_scale))
    resized_img = cv2.resize(rgb_img, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    left = (resized_w - input_w) // 2
    top = (resized_h - input_h) // 2
    cropped_img = resized_img[top:top + input_h, left:left + input_w]
    normalized_img = (cropped_img.astype(np.float32) - MEAN) / STD

    if tensor_type == 0:
        input_tensor = normalized_img.astype(np.float32)
    elif tensor_type in (2, 3, 4):
        raw_value = np.round(normalized_img / scale + zero_point)
        if tensor_type == 2:
            input_tensor = np.clip(raw_value, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_value, 0, 255).astype(np.uint8)
        else:
            input_tensor = np.clip(raw_value, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    return np.expand_dims(input_tensor, axis=0)


def preprocess_text(tokenizer, text, max_len):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
    return encoding["input_ids"].astype(np.int64)


def compute_image_embedding(vision_amlnn, image_path, input_shape, scale, zero_point, tensor_type):
    input_tensor = preprocess_image(image_path, input_shape, scale, zero_point, tensor_type)
    outputs = vision_amlnn.inference(inputs=[input_tensor], inputs_data_format="NHWC", outputs_data_format="NHWC")
    features = np.asarray(outputs[0], dtype=np.float32).reshape(1, -1)
    return l2_normalize(features, axis=1)


def compute_text_embedding(text_amlnn, tokenizer, text, input_shape):
    input_ids = preprocess_text(tokenizer, text, input_shape[-1]).reshape(input_shape)
    outputs = text_amlnn.inference(inputs=[input_ids], inputs_data_format="NHWC", outputs_data_format="NHWC")
    features = np.asarray(outputs[0], dtype=np.float32).reshape(1, -1)
    return l2_normalize(features, axis=1)


def compute_text_embeddings_batch(text_amlnn, tokenizer, texts, input_shape):
    embeddings = []
    for text in texts:
        embeddings.append(compute_text_embedding(text_amlnn, tokenizer, text, input_shape)[0])
    return np.stack(embeddings, axis=0)


def compute_similarity(image_embedding, text_embeddings, logit_scale):
    similarities = text_embeddings @ image_embedding[0]
    logits = similarities * logit_scale
    probabilities = softmax(logits, axis=0)
    return similarities, logits, probabilities


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Image-Text Matching Demo",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--vision-model", required=True, help="Path to vision .adla model")
    parser.add_argument("--text-model", required=True, help="Path to text .adla model")
    parser.add_argument("--tokenizer-dir", required=True, help="Path to CLIPTokenizer directory")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument(
        "--texts",
        nargs="+",
        required=True,
        help=(
            "Text descriptions to compare against each image.\n"
            "Separate descriptions with spaces and wrap multi-word descriptions in quotes.\n"
            "Example:\n"
            '  --texts "a red handbag" "a blue jacket" "a red bus"'
        )
    )
    parser.add_argument("--logit-scale", type=float, default=100.0, help="Logit scale factor")
    args = parser.parse_args()

    print(f"Loading CLIPTokenizer from: {args.tokenizer_dir}")
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_dir)

    vision_amlnn = AMLNN()
    vision_amlnn.init_runtime(mode="native", enable_perf=True)
    vision_amlnn.load_model(path=args.vision_model)
    vision_tensor_info = vision_amlnn.get_tensor_info()

    text_amlnn = AMLNN()
    text_amlnn.init_runtime(mode="native", enable_perf=True)
    text_amlnn.load_model(path=args.text_model)
    text_tensor_info = text_amlnn.get_tensor_info()

    print(vision_amlnn.get_sdk_version())

    vision_attr = vision_tensor_info["inputs"][0]
    vision_input_h = int(vision_attr["dims"][1])
    vision_input_w = int(vision_attr["dims"][2])
    vision_input_shape = (vision_input_h, vision_input_w)
    vision_scale = float(vision_attr["scale"])
    vision_zero_point = int(vision_attr["zp"])
    vision_tensor_type = int(vision_attr["type"])

    text_attr = text_tensor_info["inputs"][0]
    text_input_shape = tuple(int(value) for value in text_attr["dims"])
    if text_input_shape[-1] != MAX_TEXT_LENGTH:
        raise ValueError(f"CLIP text model expects sequence length {text_input_shape[-1]}, but MAX_TEXT_LENGTH is {MAX_TEXT_LENGTH}")

    text_embeddings = compute_text_embeddings_batch(text_amlnn, tokenizer, args.texts, text_input_shape)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    for text_idx, text in enumerate(args.texts):
        embedding = text_embeddings[text_idx]
        print(f"Text {text_idx}: '{text}', norm={np.linalg.norm(embedding):.6f}, min={embedding.min():.6f}, max={embedding.max():.6f}")

    image_files = []
    for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension.upper())))
    image_files.sort()

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        vision_amlnn.uninit()
        text_amlnn.uninit()
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")
    for image_file in image_files:
        print(f"  - {os.path.basename(image_file)}")
    print()

    for image_idx, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {image_idx}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            image_embedding = compute_image_embedding(vision_amlnn, image_path, vision_input_shape, vision_scale, vision_zero_point, vision_tensor_type)
            similarities, logits, probabilities = compute_similarity(image_embedding, text_embeddings, args.logit_scale)
            sorted_indices = np.argsort(probabilities)[::-1]

            print(f"Image embedding shape: {image_embedding.shape}")
            for rank, text_idx in enumerate(sorted_indices, 1):
                print(f"  {rank}. probability={probabilities[text_idx]:.6f}, similarity={similarities[text_idx]:.6f}, text='{args.texts[text_idx]}'")
        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")

        print()

    print("=" * 60)
    print("Vision model performance:")
    print(vision_amlnn.get_perf_info())
    print("Text model performance:")
    print(text_amlnn.get_perf_info())
    vision_amlnn.perf_visualize()
    text_amlnn.perf_visualize()
    vision_amlnn.uninit()
    text_amlnn.uninit()
    return 0

if __name__ == "__main__":
    main()
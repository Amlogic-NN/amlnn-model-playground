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
from pathlib import Path
from transformers import AutoTokenizer
from amlnn.api import AMLNN


LOGIT_SCALE = 4.724453449249268
LOGIT_BIAS = -16.771724700927734

MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
STD = np.array([127.5, 127.5, 127.5], dtype=np.float32)


def get_image_files(image_dir):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    return sorted(set(image_files))


def quantize_tensor(tensor, tensor_attr):
    tensor_type = int(tensor_attr["type"])

    if tensor_type == 0: # FP32 & FP16
        return tensor.astype(np.float32)

    if tensor_type not in (2, 3, 4):
        return tensor

    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])

    if s == 0.0:
        raise ValueError("Quantized input scale cannot be zero")

    raw_val = np.round((tensor.astype(np.float32) / s) + zp)

    if tensor_type == 2:    # Int8
        return np.clip(raw_val, -128, 127).astype(np.int8)
    if tensor_type == 3:    # Uint8
        return np.clip(raw_val, 0, 255).astype(np.uint8)

    return np.clip(raw_val, -32768, 32767).astype(np.int16)


def preprocess_image(image_path, tensor_attr):
    input_dims = tuple(int(dim) for dim in tensor_attr["dims"])
    input_h = input_dims[1]
    input_w = input_dims[2]

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"can't read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = (image - MEAN) / STD
    image = np.expand_dims(image, axis=0)

    if image.size != int(np.prod(input_dims)):
        raise ValueError(
            f"Vision input size mismatch: preprocessed shape={image.shape}, "
            f"ADLA shape={input_dims}"
        )

    image = image.reshape(input_dims)
    return quantize_tensor(image, tensor_attr)


def prepare_text_input(input_ids, tensor_attr):
    input_dims = tuple(int(dim) for dim in tensor_attr["dims"])
    input_ids = np.asarray(input_ids)

    if input_ids.size != int(np.prod(input_dims)):
        raise ValueError(
            f"Text input size mismatch: tokenizer shape={input_ids.shape}, "
            f"ADLA shape={input_dims}"
        )

    input_ids = input_ids.reshape(input_dims)

    # The exported text model input is int64.
    return input_ids.astype(np.int64)


def sigmoid(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def encode_labels(text_amlnn, text_tensor_attr, tokenizer, labels, text_length, prompt_template):
    text_embeddings = []

    for label in labels:
        prompt = prompt_template.format(label)
        encoded = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=text_length,
            return_tensors="np"
        )

        input_tensor = prepare_text_input(encoded["input_ids"], text_tensor_attr)
        outputs = text_amlnn.inference(inputs=[input_tensor])
        text_embeddings.append(np.asarray(outputs[0], dtype=np.float32).reshape(-1))

    return np.stack(text_embeddings, axis=0)


def postprocess(image_embedding, text_embeddings, labels, top_k):
    logits = text_embeddings @ image_embedding
    logits = logits * np.exp(LOGIT_SCALE) + LOGIT_BIAS
    probabilities = sigmoid(logits)
    indices = np.argsort(probabilities)[::-1][:min(top_k, len(labels))]

    return [
        {
            "label": labels[index],
            "probability": float(probabilities[index]),
            "logit": float(logits[index])
        }
        for index in indices
    ]


def main():
    parser = argparse.ArgumentParser(description="SigLIP2 Demo")
    parser.add_argument("--vision", required=True, help="Path to vision encoder .adla model")
    parser.add_argument("--text", required=True, help="Path to text encoder .adla model")
    parser.add_argument("--tokenizer", required=True, help="Directory containing tokenizer.model and tokenizer config files")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--prompt", nargs="+", required=True, help="Candidate labels")
    parser.add_argument("--prompt-template", default="This is a photo of {}.", help="Prompt template containing {}")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to print")
    args = parser.parse_args()

    if "{}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain {}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False
    )

    vision_amlnn = AMLNN()
    text_amlnn = AMLNN()

    vision_amlnn.init_runtime(mode="native", enable_perf=True)
    text_amlnn.init_runtime(mode="native", enable_perf=True)

    vision_amlnn.load_model(path=args.vision)
    text_amlnn.load_model(path=args.text)

    vision_tensor_info = vision_amlnn.get_tensor_info()
    text_tensor_info = text_amlnn.get_tensor_info()

    if len(vision_tensor_info["inputs"]) != 1:
        raise ValueError(f"Expected 1 vision input, got {len(vision_tensor_info['inputs'])}")
    if len(vision_tensor_info["outputs"]) != 1:
        raise ValueError(f"Expected 1 vision output, got {len(vision_tensor_info['outputs'])}")
    if len(text_tensor_info["inputs"]) != 1:
        raise ValueError(f"Expected 1 text input, got {len(text_tensor_info['inputs'])}")
    if len(text_tensor_info["outputs"]) != 1:
        raise ValueError(f"Expected 1 text output, got {len(text_tensor_info['outputs'])}")

    vision_tensor_attr = vision_tensor_info["inputs"][0]
    text_tensor_attr = text_tensor_info["inputs"][0]
    text_length = int(np.prod(text_tensor_attr["dims"]))

    print(vision_amlnn.get_sdk_version())
    print(f"Vision input: dims={vision_tensor_attr['dims']}, type={vision_tensor_attr['type']}")
    print(f"Text input: dims={text_tensor_attr['dims']}, type={text_tensor_attr['type']}")
    print(f"Text length: {text_length}")

    text_embeddings = encode_labels(
        text_amlnn,
        text_tensor_attr,
        tokenizer,
        args.prompt,
        text_length,
        args.prompt_template
    )

    image_files = get_image_files(args.image_dir)

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        vision_amlnn.uninit()
        text_amlnn.uninit()
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")
    for image_file in image_files:
        print(f"  - {os.path.basename(image_file)}")
    print()

    model_name = Path(args.vision).stem
    result_dir = f"{model_name}_result"
    os.makedirs(result_dir, exist_ok=True)

    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            input_tensor = preprocess_image(image_path, vision_tensor_attr)

            outputs = vision_amlnn.inference(inputs=[input_tensor])
            image_embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

            if image_embedding.size != text_embeddings.shape[1]:
                raise ValueError(
                    f"Embedding size mismatch: image={image_embedding.size}, "
                    f"text={text_embeddings.shape[1]}"
                )

            results = postprocess(
                image_embedding,
                text_embeddings,
                args.prompt,
                args.top_k
            )

            for rank, result in enumerate(results, 1):
                print(
                    f"    {rank}. {result['label']}: "
                    f"probability={result['probability']:.6f}, "
                    f"logit={result['logit']:.6f}"
                )

            image_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{image_name}_result.txt")

            with open(save_path, "w", encoding="utf-8") as f:
                for rank, result in enumerate(results, 1):
                    f.write(
                        f"{rank}. {result['label']}: "
                        f"probability={result['probability']:.6f}, "
                        f"logit={result['logit']:.6f}\n"
                    )

            print(f"    Result saved to: {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print(f"=" * 60)
    print("Vision model performance:")
    print(vision_amlnn.get_perf_info())
    print("Text model performance:")
    print(text_amlnn.get_perf_info())

    # vision_amlnn.perf_visualize()
    # text_amlnn.perf_visualize()

    vision_amlnn.uninit()
    text_amlnn.uninit()


if __name__ == "__main__":
    main()
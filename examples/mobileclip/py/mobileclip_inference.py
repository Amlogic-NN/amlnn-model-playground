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

"""MobileCLIP-S2 image-text matching demo using AMLNN ADLA models."""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from amlnn.api import AMLNN
from PIL import Image

from clip_tokenizer import CLIPTokenizer

IMAGE_SIZE = 256

# AMLNN tensor type ids (aligned with nnsdk2.h)
AMLNN_TENSOR_FLOAT32 = 0
AMLNN_TENSOR_INT8 = 2
AMLNN_TENSOR_UINT8 = 3
AMLNN_TENSOR_INT16 = 4
AMLNN_TENSOR_INT64 = 8


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def preprocess_image(image_path: str | Path, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize shortest edge, center crop, and convert to NCHW float32 in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = target_size / min(width, height)
    new_w = max(target_size, round(width * scale))
    new_h = max(target_size, round(height * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    image = image.crop((left, top, left + target_size, top + target_size))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def preprocess_text(tokenizer: CLIPTokenizer, text: str, max_len: int = 77) -> np.ndarray:
    tokens = tokenizer.encode(text, max_len=max_len)
    return np.asarray(tokens, dtype=np.int64).reshape(1, max_len)


def get_io_format(tensor_attr: dict) -> str:
    fmt = str(tensor_attr.get("format_name", "NCHW")).upper()
    return fmt if fmt in ("NCHW", "NHWC") else "NCHW"


def quantize_input(tensor: np.ndarray, tensor_attr: dict) -> np.ndarray:
    tensor_type = int(tensor_attr.get("type", AMLNN_TENSOR_FLOAT32))
    scale = float(tensor_attr.get("scale", 1.0) or 1.0)
    zero_point = int(tensor_attr.get("zp", 0))

    if tensor_type == AMLNN_TENSOR_FLOAT32:
        return tensor.astype(np.float32)
    if tensor_type == AMLNN_TENSOR_INT64:
        return tensor.astype(np.int64)
    if tensor_type == AMLNN_TENSOR_INT8:
        q = np.round(tensor.astype(np.float32) / scale + zero_point)
        return np.clip(q, -128, 127).astype(np.int8)
    if tensor_type == AMLNN_TENSOR_UINT8:
        q = np.round(tensor.astype(np.float32) / scale + zero_point)
        return np.clip(q, 0, 255).astype(np.uint8)
    if tensor_type == AMLNN_TENSOR_INT16:
        q = np.round(tensor.astype(np.float32) / scale + zero_point)
        return np.clip(q, -32768, 32767).astype(np.int16)

    raise ValueError(f"Unsupported input tensor type: {tensor_type}")


def run_inference(amlnn: AMLNN, model_input: np.ndarray, input_attr: dict) -> np.ndarray:
    data_format = get_io_format(input_attr)
    outputs = amlnn.inference(
        inputs=[model_input],
        inputs_data_format=data_format,
        outputs_data_format=data_format,
    )
    return outputs[0].astype(np.float32)


def compute_image_embedding(
    vision_amlnn: AMLNN,
    image_path: str | Path,
    input_attr: dict,
) -> np.ndarray:
    image_input = preprocess_image(image_path)
    model_input = quantize_input(image_input, input_attr)
    feats = run_inference(vision_amlnn, model_input, input_attr)
    feats = feats.reshape(1, -1)
    return l2_normalize(feats, axis=1)


def compute_text_embedding(
    text_amlnn: AMLNN,
    tokenizer: CLIPTokenizer,
    text: str,
    input_attr: dict,
    max_len: int,
) -> np.ndarray:
    token_input = preprocess_text(tokenizer, text, max_len=max_len)
    model_input = quantize_input(token_input, input_attr)
    feats = run_inference(text_amlnn, model_input, input_attr)
    feats = feats.reshape(1, -1)
    return l2_normalize(feats, axis=1)


def compute_text_embeddings_batch(
    text_amlnn: AMLNN,
    tokenizer: CLIPTokenizer,
    texts: list[str],
    input_attr: dict,
    max_len: int,
) -> np.ndarray:
    embeddings = []
    for text in texts:
        emb = compute_text_embedding(
            text_amlnn, tokenizer, text, input_attr, max_len=max_len
        )
        embeddings.append(emb[0])
    return np.stack(embeddings, axis=0)


def compute_similarity(
    image_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sims = text_embeddings @ image_embedding[0]
    logits = sims * logit_scale
    probs = softmax(logits, axis=0)
    return sims, logits, probs


def init_model(model_path: str, runtime_mode: str, enable_perf: bool) -> tuple[AMLNN, dict]:
    amlnn = AMLNN()
    ret = amlnn.init_runtime(mode=runtime_mode, enable_perf=enable_perf)
    if ret is not None and ret != 0:
        raise RuntimeError(f"init_runtime failed for {model_path}, ret={ret}")

    ret = amlnn.load_model(path=model_path)
    if ret is not None and ret != 0:
        raise RuntimeError(f"load_model failed for {model_path}, ret={ret}")

    tensor_info = amlnn.get_tensor_info()
    if not tensor_info or "inputs" not in tensor_info or not tensor_info["inputs"]:
        raise RuntimeError(f"Failed to query input tensor info for {model_path}")

    return amlnn, tensor_info


def print_tensor_info(prefix: str, tensor_info: dict) -> None:
    inputs = tensor_info.get("inputs", [])
    outputs = tensor_info.get("outputs", [])
    print(f"[Info] {prefix} inputs: {len(inputs)}, outputs: {len(outputs)}")
    if inputs:
        attr = inputs[0]
        print(
            f"[Info] {prefix} input[0]: name={attr.get('name')}, "
            f"dims={attr.get('dims')}, format={attr.get('format_name')}, "
            f"type={attr.get('type_name')}, scale={attr.get('scale')}, zp={attr.get('zp')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MobileCLIP-S2 image-text matching demo using AMLNN"
    )
    parser.add_argument("--vision-model", required=True, help="Path to vision .adla model")
    parser.add_argument("--text-model", required=True, help="Path to text .adla model")
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Directory containing vocab.json and merges.txt",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to input image (optional, will prompt if not provided)",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=None,
        help="Text descriptions to compare (space-separated)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=77,
        help="Maximum token sequence length (default: 77)",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=100.0,
        help="Logit scale factor (default: 100.0)",
    )
    parser.add_argument(
        "--runtime-mode",
        default="native",
        choices=["native", "nnserver"],
        help="AMLNN runtime mode (default: native)",
    )
    parser.add_argument(
        "--enable-perf",
        action="store_true",
        help="Enable AMLNN performance collection",
    )
    args = parser.parse_args()

    if args.max_len != 77:
        print(
            "[Warn] MobileCLIP-S2 text encoder is exported with fixed length 77; "
            f"--max-len={args.max_len} is ignored.",
            file=sys.stderr,
        )

    for model_path in (args.vision_model, args.text_model):
        if not os.path.exists(model_path):
            print(f"[Error] Model not found: {model_path}", file=sys.stderr)
            return 1

    print(f"[Info] Loading tokenizer from: {args.tokenizer_dir}")
    tokenizer = CLIPTokenizer()
    try:
        tokenizer.load_from_dir(args.tokenizer_dir)
    except (FileNotFoundError, OSError) as exc:
        print(f"[Error] Failed to load tokenizer: {exc}", file=sys.stderr)
        return 1

    vision_amlnn = None
    text_amlnn = None

    try:
        print(f"[Info] Loading vision model: {args.vision_model}")
        vision_amlnn, vision_tensor_info = init_model(
            args.vision_model, args.runtime_mode, args.enable_perf
        )
        print_tensor_info("Vision", vision_tensor_info)

        print(f"[Info] Loading text model: {args.text_model}")
        text_amlnn, text_tensor_info = init_model(
            args.text_model, args.runtime_mode, args.enable_perf
        )
        print_tensor_info("Text", text_tensor_info)
        print("[Info] Models initialized successfully.\n")

        vision_input_attr = vision_tensor_info["inputs"][0]
        text_input_attr = text_tensor_info["inputs"][0]

        image_path_arg = args.image_path
        texts_arg = args.texts

        while True:
            if image_path_arg:
                image_path = image_path_arg
                image_path_arg = None
            else:
                print("=" * 60)
                print("[Info] Image path (or 'exit' to quit):")
                image_path = input().strip()

            if image_path.lower() == "exit":
                print("[Info] Exiting...")
                break

            if not image_path:
                print("[Warning] Please enter an image path.")
                continue

            if not os.path.exists(image_path):
                print(f"[Error] Image not found: {image_path}")
                continue

            if texts_arg:
                texts = texts_arg
                texts_arg = None
            else:
                print(
                    "[Info] Enter text descriptions (comma-separated, "
                    "or 'skip' to use defaults):"
                )
                text_input = input().strip()
                if text_input.lower() == "skip" or not text_input:
                    texts = ["a red bus", "a red handbag", "a blue jacket"]
                    print(f"[Info] Using default texts: {texts}")
                else:
                    texts = [item.strip() for item in text_input.split(",") if item.strip()]

            if not texts:
                print("[Warning] No texts provided.")
                continue

            try:
                print(f"\n[Info] Processing image: {image_path}")
                image_embedding = compute_image_embedding(
                    vision_amlnn, image_path, vision_input_attr
                )
                print(f"[Info] Image embedding shape: {image_embedding.shape}")

                print(f"[Info] Processing {len(texts)} text(s)...")
                text_embeddings = compute_text_embeddings_batch(
                    text_amlnn,
                    tokenizer,
                    texts,
                    text_input_attr,
                    max_len=args.max_len,
                )
                print(f"[Info] Text embeddings shape: {text_embeddings.shape}")

                sims, logits, probs = compute_similarity(
                    image_embedding, text_embeddings, args.logit_scale
                )

                print("\n" + "=" * 60)
                print("MobileCLIP-S2 Image-Text Matching Results")
                print("=" * 60)
                print(f"Image: {image_path}")
                print(f"logit_scale: {args.logit_scale:.6f}")
                print("-" * 60)

                sorted_indices = np.argsort(probs)[::-1]
                for rank, idx in enumerate(sorted_indices):
                    print(
                        f"[{rank + 1}] prob={probs[idx]:.6f}  "
                        f"sim={float(sims[idx]):.6f}  text='{texts[idx]}'"
                    )
                print("=" * 60 + "\n")

                if args.enable_perf:
                    vision_perf = vision_amlnn.get_perf_info()
                    text_perf = text_amlnn.get_perf_info()
                    if vision_perf:
                        print(f"[Perf][Vision] {vision_perf}")
                    if text_perf:
                        print(f"[Perf][Text] {text_perf}")

            except Exception as exc:
                print(f"[Error] Processing failed: {exc}")
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user. Exiting...")
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if vision_amlnn is not None:
            vision_amlnn.uninit()
        if text_amlnn is not None:
            text_amlnn.uninit()

    print("[Info] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

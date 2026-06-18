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

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPTokenizer
from amlnn.api import AMLNN

MEAN = np.array([122.7709383, 116.7460125, 104.09373615], dtype=np.float32)
STD = np.array([68.5005327, 66.6321579, 70.32316305], dtype=np.float32)

# ==================== Utility Functions ====================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for array x."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize array x along specified axis."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

# ==================== Vision Preprocessing ====================
def preprocess_image(img_path, target_size=224, data_format='NHWC', tensor_type=2):
    """
    CLIP Preprocess:
    1. Scale shorter side to target_size (Bicubic).
    2. Center crop to target_size x target_size.
    3. Normalize using pre-multiplied mean/std for [0, 255] range.
    4. Format layout and quantize.
    """
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_img.shape[:2]

    # 1. Scale the shorter side
    scale = target_size / min(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # CLIP typically expects BICUBIC interpolation
    resized_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    cropped_img = resized_img[top:top+target_size, left:left+target_size]

    # 3. Normalization
    img_float = cropped_img.astype(np.float32)
    normalized_img = (img_float - MEAN) / STD

    # 4. Layout Formatting
    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    # 5. Quantization
    if tensor_type == 2:   # INT8
        input_tensor = np.clip(input_tensor, -128, 127).astype(np.int8)
    elif tensor_type == 3: # UINT8
        input_tensor = np.clip(input_tensor, 0, 255).astype(np.uint8)
    elif tensor_type == 4: # INT16
        input_tensor = np.clip(input_tensor, -32768, 32767).astype(np.int16)
    elif tensor_type == 0: # FLOAT32
        input_tensor = input_tensor.astype(np.float32)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}.")

    return input_tensor

# ==================== Text Preprocessing ====================

def preprocess_text(tokenizer: CLIPTokenizer, text: str, max_len: int = 64) -> np.ndarray:
    """
    Preprocess text for CLIP model using CLIPTokenizer.

    Args:
        tokenizer: CLIPTokenizer instance
        text (str): Input text string
        max_len (int): Maximum sequence length (default: 64)

    Returns:
        np.ndarray: Tokenized text with shape (1, max_len) as int64
    """
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    # text model input: int64[1, max_len]
    input_ids = enc["input_ids"].astype(np.int64)
    return input_ids

# ==================== Model Inference ====================

def compute_image_embedding(vision_amlnn: AMLNN, image_path: str, tensor_type=0) -> np.ndarray:
    """
    Compute image embedding using vision model.

    Args:
        vision_amlnn: AMLNN instance for vision model
        image_path (str): Path to input image

    Returns:
        np.ndarray: L2-normalized image embedding with shape (1, embed_dim)
    """
    input_data = preprocess_image(image_path, data_format ="NHWC", tensor_type=tensor_type)  # [1, 224, 224, 3]

    outputs = vision_amlnn.inference(
        inputs=[input_data],
        inputs_data_format='NHWC',
        outputs_data_format='NHWC'
    )

    feats = outputs[0].astype(np.float32)
    feats = feats.reshape(1, -1)  # Squeeze to [1, embed_dim]
    return l2_normalize(feats, axis=1)

def compute_text_embedding(text_amlnn: AMLNN, tokenizer: CLIPTokenizer, text: str, max_len: int = 64) -> np.ndarray:
    """
    Compute text embedding using text model.

    Args:
        text_amlnn: AMLNN instance for text model
        tokenizer: CLIPTokenizer instance
        text (str): Input text string
        max_len (int): Maximum sequence length

    Returns:
        np.ndarray: L2-normalized text embedding with shape (1, embed_dim)
    """
    input_ids = preprocess_text(tokenizer, text, max_len)  # [1, max_len]

    # AMLNN requires 4D input, reshape to (1, 1, 1, max_len)
    input_ids_4d = input_ids[:, None, None, :]  # [1, 1, 1, max_len]

    outputs = text_amlnn.inference(
        inputs=[input_ids_4d],
        inputs_data_format='NHWC',
        outputs_data_format='NHWC'
    )

    feats = outputs[0].astype(np.float32)
    feats = feats.reshape(1, -1)  # Squeeze to [1, embed_dim]
    return l2_normalize(feats, axis=1)

def compute_text_embeddings_batch(text_amlnn: AMLNN, tokenizer: CLIPTokenizer, texts: list, max_len: int = 64) -> np.ndarray:
    """
    Compute text embeddings for multiple texts.

    Args:
        text_amlnn: AMLNN instance for text model
        tokenizer: CLIPTokenizer instance
        texts (list): List of input text strings
        max_len (int): Maximum sequence length

    Returns:
        np.ndarray: L2-normalized text embeddings with shape (num_texts, embed_dim)
    """
    embeddings = []
    for text in texts:
        emb = compute_text_embedding(text_amlnn, tokenizer, text, max_len)
        embeddings.append(emb[0])  # Remove batch dimension
    return np.stack(embeddings, axis=0)  # [num_texts, embed_dim]

# ==================== Similarity Calculation ====================

def compute_similarity(image_embedding: np.ndarray, text_embeddings: np.ndarray, logit_scale: float = 100.0) -> tuple:
    """
    Compute similarity between image and text embeddings.

    Args:
        image_embedding (np.ndarray): Image embedding with shape (1, embed_dim)
        text_embeddings (np.ndarray): Text embeddings with shape (num_texts, embed_dim)
        logit_scale (float): Scale factor for logits

    Returns:
        tuple: (similarities, logits, probabilities)
    """
    # Cosine similarity (embeddings are already L2-normalized)
    sims = text_embeddings @ image_embedding[0]  # [num_texts]
    logits = sims * logit_scale  # [num_texts]
    probs = softmax(logits, axis=0)  # [num_texts]

    return sims, logits, probs

# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description='CLIP Image-Text Matching Demo using AMLNN')
    parser.add_argument('--vision-model-path', required=True, help='Path to vision model')
    parser.add_argument('--text-model-path', required=True, help='Path to text model')
    parser.add_argument('--tokenizer-dir', required=True, help='Path to CLIPTokenizer directory')
    parser.add_argument('--image-path', default=None, help='Path to input image (optional, will prompt if not provided)')
    parser.add_argument('--texts', nargs='+', default=None, help='List of text descriptions to compare')
    parser.add_argument('--max-len', type=int, default=64, help='Maximum token sequence length (default: 64)')
    parser.add_argument('--logit-scale', type=float, default=100.0, help='Logit scale factor (default: 100.0)')

    args = parser.parse_args()

    # Validate model paths
    # if not os.path.exists(args.vision_model_path):
    #     print(f"[Error] Vision model not found: {args.vision_model_path}")
    #     return -1

    # if not os.path.exists(args.text_model_path):
    #     print(f"[Error] Text model not found: {args.text_model_path}")
    #     return -1

    # Load tokenizer
    print(f"[Info] Loading CLIPTokenizer from: {args.tokenizer_dir}")
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_dir)

    # Initialize vision model
    # print(f"[Info] Initializing vision model: {args.vision_model_path}")
    vision_amlnn = AMLNN()

    vision_amlnn.init_runtime(mode="native", enable_perf=True)

    vision_amlnn.load_model(path=args.vision_model_path)

    vision_tensor_info = vision_amlnn.get_tensor_info()


    # Initialize text model
    # print(f"[Info] Initializing text model: {args.text_model_path}")
    text_amlnn = AMLNN()

    text_amlnn.init_runtime(mode="native", enable_perf=True)

    text_amlnn.load_model(path=args.text_model_path)

    text_tensor_info = text_amlnn.get_tensor_info()

    print("[Info] Models initialized successfully.\n")

    try:
        # Interactive loop
        while True:
            # Get image path
            if args.image_path:
                image_path = args.image_path
                args.image_path = None  # Clear for next iteration
            else:
                print("=" * 60)
                print("[Info] Image Path (or 'exit' to quit):")
                image_path = input().strip()

            # Check for exit
            if image_path.lower() == 'exit':
                print("[Info] Exiting...")
                break

            # Validate image path
            if not image_path:
                print("[Warning] Please enter an image path.")
                continue

            if not os.path.exists(image_path):
                print(f"[Error] Image not found: {image_path}")
                continue

            # Get texts to compare
            if args.texts:
                texts = args.texts
                args.texts = None  # Clear for next iteration
            else:
                print("[Info] Enter text descriptions (comma-separated, or 'skip' to use defaults):")
                text_input = input().strip()

                if text_input.lower() == 'skip' or not text_input:
                    # Default texts for demo
                    texts = [
                        "a red handbag",
                        "a blue jacket",
                        "a red bus",
                    ]
                    print(f"[Info] Using default texts: {texts}")
                else:
                    texts = [t.strip() for t in text_input.split(',') if t.strip()]

            if not texts:
                print("[Warning] No texts provided.")
                continue

            try:
                # Compute image embedding
                vision_tensor_attr = vision_tensor_info["inputs"][0]
                vision_tensor_type = int(vision_tensor_attr["type"])
                print(f"\n[Info] Processing image: {image_path}")
                image_embedding = compute_image_embedding(vision_amlnn, image_path, tensor_type=vision_tensor_type)
                print(f"[Info] Image embedding shape: {image_embedding.shape}")

                # Compute text embeddings
                print(f"[Info] Processing {len(texts)} text(s)...")
                text_embeddings = compute_text_embeddings_batch(text_amlnn, tokenizer, texts, args.max_len)
                print(f"[Info] Text embeddings shape: {text_embeddings.shape}")

                # Compute similarity
                sims, logits, probs = compute_similarity(image_embedding, text_embeddings, args.logit_scale)

                # Print results
                print("\n" + "=" * 60)
                print("CLIP Image-Text Matching Results")
                print("=" * 60)
                print(f"Image: {image_path}")
                print(f"logit_scale: {args.logit_scale:.6f}")
                print("-" * 60)

                # Sort by probability (descending)
                sorted_indices = np.argsort(probs)[::-1]
                for rank, i in enumerate(sorted_indices):
                    print(f"[{rank + 1}] prob={probs[i]:.6f}  sim={float(sims[i]):.6f}  text='{texts[i]}'")

                print("=" * 60 + "\n")

            except Exception as e:
                print(f"[Error] Processing failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        print("\n\n[Info] Interrupted by user. Exiting...")

    finally:
        # Cleanup
        vision_amlnn.uninit()
        text_amlnn.uninit()

    print("[Info] Done.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

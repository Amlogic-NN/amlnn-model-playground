# -*- coding: utf-8 -*-
"""
Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.

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

# This inference script is designed for CLIP model using AMLNNLite.

import os
import argparse
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer
from amlnnlite.api import AMLNNLite

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

def preprocess_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """
    Preprocess image for CLIP model.

    Args:
        image_path (str): Path to input image
        target_size (int): Target image size (default: 224)

    Returns:
        np.ndarray: Preprocessed image data with shape (1, target_size, target_size, 3) in NHWC format
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Scale the shorter side
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image_resized = image.resize((new_width, new_height), resample=Image.BICUBIC)

    # Center crop
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    image_cropped = image_resized.crop((left, top, right, bottom))

    # Convert to numpy array and normalize to [0, 1]
    image_np = np.array(image_cropped).astype(np.float32) / 255.0

    # CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    image_np = (image_np - mean) / std

    # Add batch dimension: HWC -> NHWC
    image_np = np.expand_dims(image_np, axis=0)

    return image_np.astype(np.float32)  # [1, 224, 224, 3]

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

def compute_image_embedding(vision_amlnn: AMLNNLite, image_path: str) -> np.ndarray:
    """
    Compute image embedding using vision model.

    Args:
        vision_amlnn: AMLNNLite instance for vision model
        image_path (str): Path to input image
    
    Returns:
        np.ndarray: L2-normalized image embedding with shape (1, embed_dim)
    """
    input_data = preprocess_image(image_path)  # [1, 224, 224, 3]

    outputs = vision_amlnn.inference(
        inputs=[input_data],
        inputs_data_format='NHWC',
        outputs_data_format='NHWC'
    )

    feats = outputs[0].astype(np.float32)
    feats = feats.reshape(1, -1)  # Squeeze to [1, embed_dim]
    return l2_normalize(feats, axis=1)

def compute_text_embedding(text_amlnn: AMLNNLite, tokenizer: CLIPTokenizer, text: str, max_len: int = 64) -> np.ndarray:
    """
    Compute text embedding using text model.

    Args:
        text_amlnn: AMLNNLite instance for text model
        tokenizer: CLIPTokenizer instance
        text (str): Input text string
        max_len (int): Maximum sequence length

    Returns:
        np.ndarray: L2-normalized text embedding with shape (1, embed_dim)
    """
    input_ids = preprocess_text(tokenizer, text, max_len)  # [1, max_len]
    print(f"input_ids: {input_ids}")

    # AMLNNLite requires 4D input, reshape to (1, 1, 1, max_len)
    input_ids_4d = input_ids[:, None, None, :]  # [1, 1, 1, max_len]

    outputs = text_amlnn.inference(
        inputs=[input_ids_4d],
        inputs_data_format='NHWC',
        outputs_data_format='NHWC'
    )

    feats = outputs[0].astype(np.float32)
    feats = feats.reshape(1, -1)  # Squeeze to [1, embed_dim]
    return l2_normalize(feats, axis=1)

def compute_text_embeddings_batch(text_amlnn: AMLNNLite, tokenizer: CLIPTokenizer, texts: list, max_len: int = 64) -> np.ndarray:
    """
    Compute text embeddings for multiple texts.

    Args:
        text_amlnn: AMLNNLite instance for text model
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
    parser = argparse.ArgumentParser(description='CLIP Image-Text Matching Demo using AMLNNLite')
    parser.add_argument('--vision-model', required=True, help='Path to vision model (.adla)')
    parser.add_argument('--text-model', required=True, help='Path to text model (.adla)')
    parser.add_argument('--tokenizer-dir', required=True, help='Path to CLIPTokenizer directory')
    parser.add_argument('--image-path', default=None, help='Path to input image (optional, will prompt if not provided)')
    parser.add_argument('--texts', nargs='+', default=None, help='List of text descriptions to compare')
    parser.add_argument('--max-len', type=int, default=64, help='Maximum token sequence length (default: 64)')
    parser.add_argument('--logit-scale', type=float, default=100.0, help='Logit scale factor (default: 100.0)')

    args = parser.parse_args()

    # Validate model paths
    if not os.path.exists(args.vision_model):
        print(f"[Error] Vision model not found: {args.vision_model}")
        return -1

    if not os.path.exists(args.text_model):
        print(f"[Error] Text model not found: {args.text_model}")
        return -1

    # Load tokenizer
    print(f"[Info] Loading CLIPTokenizer from: {args.tokenizer_dir}")
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_dir)

    # Initialize vision model
    print(f"[Info] Initializing vision model: {args.vision_model}")
    vision_amlnn = AMLNNLite()
    vision_amlnn.config(model_path=args.vision_model, run_cycles=1)
    vision_amlnn.init()

    # Initialize text model
    print(f"[Info] Initializing text model: {args.text_model}")
    text_amlnn = AMLNNLite()
    text_amlnn.config(model_path=args.text_model, run_cycles=1)
    text_amlnn.init()

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
                print(f"\n[Info] Processing image: {image_path}")
                image_embedding = compute_image_embedding(vision_amlnn, image_path)
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

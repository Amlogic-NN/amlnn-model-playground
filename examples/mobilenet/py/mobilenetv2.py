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
import time
from PIL import Image
from amlnnlite.api import AMLNNLite


def preprocess(image_path: str) -> np.ndarray:
    """
    Preprocess the input image for MobileNetV2 quantized model.

    Steps:
        1. Load image and convert to RGB
        2. Resize to 224x224
        3. Normalize to [-1, 1]
        4. Quantize to uint8 with zero-point = 128, scale = 0.0078125

    Args:
        image_path (str): Path to input image

    Returns:
        np.ndarray: Preprocessed image data with shape (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.array(img, dtype=np.float32)

    # Normalize to [-1, 1]
    img = img / 127.5 - 1.0

    # Expand batch dimension
    data = np.expand_dims(img, axis=0)

    # Quantization (uint8)
    data = data / 0.0078125 + 128
    data = np.clip(data, 0, 255).astype(np.uint8)

    return data


def postprocess_topk(predictions: np.ndarray, labels_path: str, k: int = 5, use_softmax: bool = True) -> None:
    """
    Postprocess model output and print top-K classification results.

    Args:
        predictions (np.ndarray): Raw model output (logits)
        labels_path (str): Path to labels.txt
        k (int): Number of top results to display
        use_softmax (bool): If True, apply softmax to convert logits to probabilities
    """
    predictions = predictions.reshape(-1).astype(np.float32)

    if use_softmax:
        exp_predictions = np.exp(predictions - np.max(predictions))
        probabilities = exp_predictions / np.sum(exp_predictions)
        scores = probabilities
        score_label = "probability"
    else:
        scores = predictions
        score_label = "score"

    # Load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    # Get top-k indices based on scores
    top_indices = np.argsort(scores)[::-1][:k]

    print(f"\nTop-{k} Classification Results:")
    for rank, idx in enumerate(top_indices, start=1):
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        score = scores[idx]
        if use_softmax:
            print(f"  {rank}. {label:<25} ({score_label}: {score:.4f})")
        else:
            print(f"  {rank}. {label:<25} ({score_label}: {score:.6f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='./mobilenet_v2_1.0_224_quant.tflite')
    parser.add_argument('--run-cycles', type=int, default=1, help='Number of inference cycles to run (for performance testing)')
    args = parser.parse_args()
    
    # Initialize AMLNNLite
    amlnn = AMLNNLite()
    amlnn.config(
        model_path=args.model_path           # Model file path, Support ADLD and quantized TFlite models
    )
    amlnn.init()

    # Find all image files in the 01_export_model directory
    image_dir = "./"
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print("No image files found in", image_dir)
        amlnn.uninit()
        return
    
    print(f"Found {len(image_files)} image files to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)
        
        try:
            # Preprocess input
            input_data = preprocess(image_path)

            # Run inference for specified cycles
            inference_times = []
            outputs = None
            for cycle in range(args.run_cycles):
                start_time = time.time()
                outputs = amlnn.inference(
                    inputs=[input_data]
                )
                inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                inference_times.append(inference_time)
            
            # Print performance statistics if running multiple cycles
            if args.run_cycles > 1:
                avg_time = np.mean(inference_times)
                min_time = np.min(inference_times)
                max_time = np.max(inference_times)
                print(f"\nInference Performance ({args.run_cycles} cycles):")
                print(f"  Average: {avg_time:.2f} ms")
                print(f"  Min: {min_time:.2f} ms")
                print(f"  Max: {max_time:.2f} ms")
            
            # Postprocess results (only show results from last inference)
            postprocess_topk(outputs[0], "./labels.txt", k=5, use_softmax=True)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        
        print()

    # Optional visualization
    amlnn.visualize()

    # Release resources
    amlnn.uninit()


if __name__ == "__main__":
    main()

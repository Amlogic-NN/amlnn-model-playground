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

import argparse
import shutil
from pathlib import Path

import numpy as np
from amlnn.api import AMLNN

MEAN = np.array([122.7709383, 116.7460125, 104.09373615], dtype=np.float32)
STD = np.array([68.5005327, 66.6321579, 70.32316305], dtype=np.float32)


def snapshot_adla_files(search_dirs):
    files = {}

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue

        for path in search_dir.rglob("*.adla"):
            stat = path.stat()
            files[path.resolve()] = (stat.st_mtime_ns, stat.st_size)

    return files


def find_updated_adla_files(search_dirs, known_files):
    current_files = snapshot_adla_files(search_dirs)
    updated_files = [path for path, state in current_files.items() if known_files.get(path) != state]
    return sorted(updated_files, key=lambda path: current_files[path][0], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Export ONNX to ADLA")
    parser.add_argument("--text-onnx", required=True, help="Path to Text ONNX model")
    parser.add_argument("--image-onnx", required=True, help="Path to Image ONNX model")
    parser.add_argument("--text-dataset-path", help="Path to a `.txt` containing all the paths to the quantization inputs for the text model (Not needed if using FP16).")
    parser.add_argument("--image-dataset-path", help="Path to a `.txt` containing all the paths to the quantization images for the image model (Not needed if using FP16).")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--output-dir", default="../model", help="Output directory for exported .adla models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_model_path = Path(args.image_onnx).resolve()
    image_dataset_path = Path(args.image_dataset_path).resolve() if args.image_dataset_path else None

    if not image_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {image_model_path}")

    if image_dataset_path is not None:
        if not image_dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {image_dataset_path}")

        if image_dataset_path.suffix.lower() != ".txt":
            raise ValueError(f"Dataset path must be a .txt file: {image_dataset_path}")

    search_dirs = {Path.cwd().resolve(), image_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()
    amlnn.load_onnx(
        model=str(image_model_path)
    )

    amlnn.config(
        export_intermediate=True,
        # normalization_mean=[MEAN.tolist()],
        # normalization_std=[STD.tolist()],
        quantized_dtype="w16a16",
        activation_dtype="f16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
    )

    if image_dataset_path is None:
        amlnn.compile()
    else:
        amlnn.compile(dataset=str(image_dataset_path))

    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("Image export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]
    image_output_path = output_dir / generated_path.name

    if generated_path != image_output_path:
        shutil.copy2(generated_path, image_output_path)

    if not image_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {image_output_path}")

    print(f"saved: {image_output_path}")

    text_model_path = Path(args.text_onnx).resolve()
    text_dataset_path = Path(args.text_dataset_path).resolve() if args.text_dataset_path else None

    if not text_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {text_model_path}")

    if text_dataset_path is not None:
        if not text_dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {text_dataset_path}")

        if text_dataset_path.suffix.lower() != ".txt":
            raise ValueError(f"Dataset path must be a .txt file: {text_dataset_path}")

    search_dirs = {Path.cwd().resolve(), text_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()
    amlnn.load_onnx(
        model=str(text_model_path)
    )

    amlnn.config(
        export_intermediate=True,
        # normalization_mean=[MEAN.tolist()],
        # normalization_std=[STD.tolist()],
        quantized_dtype="w16a16",
        activation_dtype="f16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
    )

    if text_dataset_path is None:
        amlnn.compile()
    else:
        amlnn.compile(dataset=str(text_dataset_path))

    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("Text export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]
    text_output_path = output_dir / generated_path.name

    if generated_path != text_output_path:
        shutil.copy2(generated_path, text_output_path)

    if not text_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {text_output_path}")

    print(f"saved: {text_output_path}")


if __name__ == "__main__":
    main()
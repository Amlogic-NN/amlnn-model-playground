# -*- coding: utf-8 -*-

"""
Copyright (C) 2024-2025 Amlogic, Inc. All rights reserved.

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

# Normalization constants used for quantization config
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


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
    parser = argparse.ArgumentParser(description="Export DINO backbone and linear classifier ONNX models to ADLA")
    parser.add_argument("--backbone-onnx", required=True, help="Path to backbone ONNX model")
    parser.add_argument("--classifier-onnx", required=True, help="Path to linear classifier ONNX model")
    parser.add_argument("--dataset-path", help="Path to quantization dataset .txt for backbone model")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--output-dir", default="../model", help="Directory where the generated .adla models will be saved")
    args = parser.parse_args()

    backbone_model_path = Path(args.backbone_onnx).resolve()
    linear_model_path = Path(args.classifier_onnx).resolve()
    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else None
    output_dir = Path(args.output_dir).resolve()

    if not backbone_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {backbone_model_path}")

    if not linear_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {linear_model_path}")

    if dataset_path is not None:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        if dataset_path.suffix.lower() != ".txt":
            raise ValueError(f"Dataset path must be a .txt file: {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export backbone
    search_dirs = {Path.cwd().resolve(), backbone_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()

    # NOTE: These node names may vary depending on your model. Please ensure the output order remains the same.
    amlnn.load_onnx(
        model=str(backbone_model_path),
        outputs=[
            "embedding_block_8",
            "embedding_block_9",
            "embedding_block_10",
            "embedding_block_11"
        ]
    )

    amlnn.config(
        normalization_mean=[MEAN.tolist()],
        normalization_std=[STD.tolist()],
        quantized_dtype="w8a8",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}",
    )

    if dataset_path is None:
        amlnn.compile()
    else:
        amlnn.compile(dataset=str(dataset_path))

    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("export_adla did not create or update a backbone .adla file")

    generated_path = updated_adla_files[0]
    output_path = output_dir / generated_path.name

    if generated_path != output_path:
        shutil.copy2(generated_path, output_path)

    if not output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {output_path}")

    print(f"saved: {output_path}")

    # Export linear classifier
    search_dirs = {Path.cwd().resolve(), linear_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()

    amlnn.load_onnx(model=str(linear_model_path))

    amlnn.config(
        quantized_dtype="w8a16",
        activation_dtype="f16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}",
    )

    amlnn.compile()
    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("export_adla did not create or update a classifier .adla file")

    generated_path = updated_adla_files[0]
    output_path = output_dir / generated_path.name

    if generated_path != output_path:
        shutil.copy2(generated_path, output_path)

    if not output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {output_path}")

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
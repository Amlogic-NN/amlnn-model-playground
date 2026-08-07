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
MEAN = np.array([0, 0, 0], dtype=np.float32)
STD  = np.array([255, 255, 255], dtype=np.float32)

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


def get_output_path(adla_arg, model_path):
    requested_path = Path(adla_arg)

    if requested_path.suffix.lower() == ".adla":
        return requested_path

    return requested_path / f"{model_path.stem}.adla"


def main():
    parser = argparse.ArgumentParser(description="Export ONNX to ADLA")
    parser.add_argument("--onnx", required=True, help="Path to input .onnx model")
    parser.add_argument("--dataset-path", help="Path to quantization dataset")
    parser.add_argument("--target-platform", required=True, help="Platform ID, for example: 001, 002, 003")
    parser.add_argument("--adla", default="../model", help="Output .adla file or directory (default: ../model)")
    args = parser.parse_args()

    model_path = Path(args.onnx).resolve()
    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else None

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if dataset_path is not None:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        if dataset_path.suffix.lower() != ".txt":
            raise ValueError(f"Dataset path must be a .txt file: {dataset_path}")

    output_path = get_output_path(args.adla, model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()

    # NOTE: These node names may vary depending on your model. Please ensure the output order remains the same.
    amlnn.load_onnx(
        model=str(model_path),
        outputs=[
            "/model.23/Reshape_3_output_0", # <-- Stride 8 cls (1x80x6400 grid)
            "/model.23/Reshape_output_0",   # <-- Stride 8 dfl (1x64x6400 grid)
            "/model.23/Reshape_4_output_0", # <-- Stride 16 cls (1x80x1600 grid)
            "/model.23/Reshape_1_output_0", # <-- Stride 16 dfl (1x64x1600 grid)
            "/model.23/Reshape_5_output_0", # <-- Stride 32 cls (1x80x400 grid)
            "/model.23/Reshape_2_output_0"  # <-- Stride 32 dfl (1x64x400 grid)
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
        raise RuntimeError("export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]

    if generated_path != output_path.resolve():
        shutil.copy2(generated_path, output_path)

    if not output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {output_path}")

    print(f"saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
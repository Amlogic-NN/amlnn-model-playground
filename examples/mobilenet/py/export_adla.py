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

# Normalization constants for ImageNet
MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)
STD  = np.array([58.395, 57.120, 57.375], dtype=np.float32)

def snapshot_adla_files(search_dir):
    return {path: path.stat().st_mtime for path in search_dir.rglob("*.adla")}


def find_updated_adla_files(search_dir, known_files):
    current_files = snapshot_adla_files(search_dir)
    updated_files = [
        path for path, mtime in current_files.items()
        if path not in known_files or mtime > known_files[path]
    ]
    return sorted(
        updated_files,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Export ONNX/TFLite to ADLA")
    parser.add_argument("--model", required=True, help="Path to input model (.onnx or .tflite)")
    parser.add_argument("--dataset-path", required=True, help="Path to quant dataset")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--adla", default="../model", help="Optional output .adla path")
    args = parser.parse_args()

    search_dir = Path.cwd()
    known_adla_files = snapshot_adla_files(search_dir) if args.adla else {}

    amlnn = AMLNN()

    ext = Path(args.model).suffix.lower()

    if ext == ".onnx":
        print(f"Loading ONNX model: {args.model}")
        amlnn.load_onnx(model=args.model)
        amlnn.config(
            normalization_mean=[MEAN.tolist()],
            normalization_std=[STD.tolist()],
            quantized_dtype="w8a8",
            activation_quant_algo="omse",
            target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}",
        )
        amlnn.compile(dataset=args.dataset_path)
    elif ext == ".tflite":
        # Already pre quantized so we do not need to quantize it again
        print(f"Loading TFLite model: {args.model}")
        amlnn.load_tflite(model=args.model, quantized_model=True)
        amlnn.config(
            quantized_dtype='w8a8', 
            target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
        )
        amlnn.compile()
    else:
        print(f"Error: Unsupported model extension '{ext}'. Must be .onnx or .tflite")
        return

    amlnn.export_adla()

    if args.adla:
        new_adla_files = find_updated_adla_files(search_dir, known_adla_files)
        if not new_adla_files:
            raise RuntimeError("export_adla did not create or update a .adla file")

        output_path = Path(args.adla)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if new_adla_files[0].resolve() != output_path.resolve():
            shutil.copy2(new_adla_files[0], output_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

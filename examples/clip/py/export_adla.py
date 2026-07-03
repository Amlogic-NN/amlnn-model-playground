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
    parser = argparse.ArgumentParser(description="Export ONNX to ADLA")
    parser.add_argument("--text-onnx", required=True, help="Path to Text ONNX model")
    parser.add_argument("--vision-onnx", required=True, help="Path to Vision ONNX model")
    parser.add_argument("--text-dataset-path", help="Path to a `.txt` containing all the paths to the quantization images for the text model (Not needed only if you are using `FP16`, required otherwise.")
    parser.add_argument("--vision-dataset-path", help="Path to a `.txt` containing all the paths to the quantization images for the vision model (Not needed only if you are using `FP16`, required otherwise.")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--adla", default="../model", help="Optional output .adla path")
    args = parser.parse_args()

    search_dir = Path.cwd()
    known_adla_files = snapshot_adla_files(search_dir) if args.adla else {}

    amlnn = AMLNN()
    amlnn.load_onnx(model=args.vision_onnx,
        outputs=[
        "image_embeds"  # <-- Vision embedding 1x512
    ])

    amlnn.config(
        normalization_mean=[MEAN.tolist()],
        normalization_std=[STD.tolist()],
        quantized_dtype='w8a16',
        activation_dtype="fp16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
    )

    # NOTE: You will have add the vision-dataset-path argument IF YOU ARE QUANTIZING TO INT8/UINT8/INT16
    # amlnn.compile(args.text_dataset_path)

    amlnn.compile()

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


    amlnn = AMLNN()
    amlnn.load_onnx(model=args.text_onnx,
        outputs=[
        "text_embeds"   # <-- Text embedding 1x512
    ])

    amlnn.config(
        quantized_dtype='w8a16',
        activation_dtype="fp16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
    )

    # NOTE: You will have add the vision-dataset-path argument IF YOU ARE QUANTIZING TO INT8/UINT8/INT16
    # amlnn.compile(args.vision_dataset_path)

    amlnn.compile()

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

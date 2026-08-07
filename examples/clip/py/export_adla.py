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


def get_output_path(adla_arg, model_path):
    requested_path = Path(adla_arg)

    if requested_path.suffix.lower() == ".adla":
        return requested_path

    return requested_path / f"{model_path.stem}.adla"


def main():
    parser = argparse.ArgumentParser(description="Export ONNX to ADLA")
    parser.add_argument("--text-onnx", required=True, help="Path to Text ONNX model")
    parser.add_argument("--vision-onnx", required=True, help="Path to Vision ONNX model")
    parser.add_argument("--text-dataset-path", help="Path to a `.txt` containing all the paths to the quantization images for the text model (Not needed only if you are using `FP16`, required otherwise.")
    parser.add_argument("--vision-dataset-path", help="Path to a `.txt` containing all the paths to the quantization images for the vision model (Not needed only if you are using `FP16`, required otherwise.")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--adla", default="../model", help="Output directory for exported .adla models")
    args = parser.parse_args()

    vision_model_path = Path(args.vision_onnx).resolve()
    vision_dataset_path = Path(args.vision_dataset_path).resolve() if args.vision_dataset_path else None

    if not vision_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {vision_model_path}")

    if vision_dataset_path is not None:
        if not vision_dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {vision_dataset_path}")

        if vision_dataset_path.suffix.lower() != ".txt":
            raise ValueError(f"Dataset path must be a .txt file: {vision_dataset_path}")

    vision_output_path = get_output_path(args.adla, vision_model_path)
    vision_output_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), vision_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()
    amlnn.load_onnx(
        model=str(vision_model_path),
        outputs=[
            "image_embeds"  # <-- Vision embedding 1x512
        ]
    )

    amlnn.config(
        normalization_mean=[MEAN.tolist()],
        normalization_std=[STD.tolist()],
        quantized_dtype="w8a16",
        activation_dtype="f16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
    )

    if vision_dataset_path is None:
        amlnn.compile()
    else:
        amlnn.compile(dataset=str(vision_dataset_path))

    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("Vision export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]

    if generated_path != vision_output_path.resolve():
        shutil.copy2(generated_path, vision_output_path)

    if not vision_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {vision_output_path}")

    print(f"saved: {vision_output_path.resolve()}")

    text_model_path = Path(args.text_onnx).resolve()
    text_dataset_path = Path(args.text_dataset_path).resolve() if args.text_dataset_path else None

    if not text_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {text_model_path}")

    if text_dataset_path is not None:
        if not text_dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {text_dataset_path}")

        if text_dataset_path.suffix.lower() != ".txt":
            raise ValueError(f"Dataset path must be a .txt file: {text_dataset_path}")

    if not text_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {text_model_path}")

    text_output_path = get_output_path(args.adla, text_model_path)
    text_output_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), text_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()
    amlnn.load_onnx(
        model=str(text_model_path),
        outputs=[
            "text_embeds"   # <-- Text embedding 1x512
        ]
    )

    amlnn.config(
        quantized_dtype="w8a16",
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

    if generated_path != text_output_path.resolve():
        shutil.copy2(generated_path, text_output_path)

    if not text_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {text_output_path}")

    print(f"saved: {text_output_path.resolve()}")


if __name__ == "__main__":
    main()
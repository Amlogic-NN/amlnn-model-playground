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
DET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DET_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
REC_MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
REC_STD  = np.array([127.5, 127.5, 127.5], dtype=np.float32)

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
    parser.add_argument("--det-onnx", required=True, help="Path to Detection ONNX model")
    parser.add_argument("--rec-onnx", required=True, help="Path to Recognition ONNX model")
    parser.add_argument("--det-dataset-path", help="Path to Detection quant dataset")
    parser.add_argument("--rec-dataset-path", help="Path to Recognition quant dataset")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--adla", default="../model", help="Optional output .adla path")
    args = parser.parse_args()

    det_model_path = Path(args.det_onnx).resolve()
    det_dataset_path = Path(args.det_dataset_path).resolve() if args.det_dataset_path else None

    if not det_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {det_model_path}")

    if not det_dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {det_dataset_path}")

    if det_dataset_path.suffix.lower() != ".txt":
        raise ValueError(f"Dataset path must be a .txt file: {det_dataset_path}")

    det_output_path = get_output_path(args.adla, det_model_path)
    det_output_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), det_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()

    amlnn.load_onnx(model=str(det_model_path))

    amlnn.config(
        normalization_mean=[DET_MEAN.tolist()],
        normalization_std=[DET_STD.tolist()],
        quantized_dtype="w8a8",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}",
    )

    if det_dataset_path is None:
        amlnn.compile()
    else:
        amlnn.compile(dataset=str(det_dataset_path))

    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("Detection export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]

    if generated_path != det_output_path.resolve():
        shutil.copy2(generated_path, det_output_path)

    if not det_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {det_output_path}")

    print(f"saved: {det_output_path.resolve()}")

    rec_model_path = Path(args.rec_onnx).resolve()
    rec_dataset_path = Path(args.rec_dataset_path).resolve() if args.rec_dataset_path else None

    if not rec_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {rec_model_path}")

    if rec_dataset_path is not None:
        if not rec_dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {rec_dataset_path}")

        if rec_dataset_path.suffix.lower() != ".txt":
            raise ValueError(f"Dataset path must be a .txt file: {rec_dataset_path}")

    rec_output_path = get_output_path(args.adla, rec_model_path)
    rec_output_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), rec_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()

    amlnn.load_onnx(model=str(rec_model_path))

    amlnn.config(
        normalization_mean=[REC_MEAN.tolist()],
        normalization_std=[REC_STD.tolist()],
        quantized_dtype="w8a16",
        activation_dtype="f16",
        quantized_method="perchannel",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}",
    )

    if rec_dataset_path is None:
        amlnn.compile()
    else:
        amlnn.compile(dataset=str(rec_dataset_path))

    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("Recognition export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]

    if generated_path != rec_output_path.resolve():
        shutil.copy2(generated_path, rec_output_path)

    if not rec_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {rec_output_path}")

    print(f"saved: {rec_output_path.resolve()}")


if __name__ == "__main__":
    main()
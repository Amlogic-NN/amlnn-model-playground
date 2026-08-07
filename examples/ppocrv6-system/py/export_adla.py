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

DET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
REC_MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
REC_STD = np.array([127.5, 127.5, 127.5], dtype=np.float32)


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


def convert_model(
    onnx_path,
    output_path,
    target_platform,
    normalization_mean,
    normalization_std,
    quantized_dtype,
    dataset_path,
    inputs,
    input_shapes,
    activation_dtype=None,
    quantized_method=None,
):
    print(f"\nConverting: {onnx_path}")

    search_dir = Path.cwd()
    known_files = snapshot_adla_files(search_dir)

    amlnn = AMLNN()
    amlnn.load_onnx(model=onnx_path, inputs=inputs, input_shapes=input_shapes)

    config_kwargs = {
        "normalization_mean": [normalization_mean.tolist()],
        "normalization_std": [normalization_std.tolist()],
        "quantized_dtype": quantized_dtype,
        "target_platform": f"PRODUCT_PID0XA{target_platform.zfill(3)}",
    }
    if activation_dtype is not None:
        config_kwargs["activation_dtype"] = activation_dtype
    if quantized_method is not None:
        config_kwargs["quantized_method"] = quantized_method

    amlnn.config(**config_kwargs)
    amlnn.compile(dataset=dataset_path)
    amlnn.export_adla()

    new_files = find_updated_adla_files(search_dir, known_files)
    if not new_files:
        raise RuntimeError("export_adla did not create or update a .adla file")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if new_files[0].resolve() != output_path.resolve():
        shutil.copy2(new_files[0], output_path)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export PP-OCRv6 ONNX models to ADLA")
    parser.add_argument("--det-onnx", required=True, help="Path to detection ONNX model")
    parser.add_argument("--rec-onnx", required=True, help="Path to recognition ONNX model")
    parser.add_argument("--det-dataset-path", required=True, help="Path to detection quant dataset")
    parser.add_argument("--rec-dataset-path", required=True, help="Path to recognition quant dataset")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--out-dir", default="../model", help="Output directory for ADLA models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    convert_model(
        args.det_onnx,
        out_dir / f"{Path(args.det_onnx).stem}.adla",
        args.target_platform,
        DET_MEAN,
        DET_STD,
        quantized_dtype="w8a16",
        dataset_path=args.det_dataset_path,
        activation_dtype="f16",
        quantized_method="perchannel",
        inputs=['x'],
        input_shapes=[[1, 3, 960, 960]],
    )

    convert_model(
        args.rec_onnx,
        out_dir / f"{Path(args.rec_onnx).stem}.adla",
        args.target_platform,
        REC_MEAN,
        REC_STD,
        quantized_dtype="w8a16",
        dataset_path=args.rec_dataset_path,
        activation_dtype="f16",
        quantized_method="perchannel",
        inputs=['x'],
        input_shapes=[[1, 3, 48, 320]],
    )


if __name__ == "__main__":
    main()

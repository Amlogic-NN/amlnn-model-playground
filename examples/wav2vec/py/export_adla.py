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
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--output-dir", default="../model", help="Directory where the generated .adla model will be saved")
    args = parser.parse_args()

    model_path = Path(args.onnx).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()

    # NOTE: These node names may be different depending on your model
    amlnn.load_onnx(model=str(model_path), outputs=[
        "output"  # <-- 1x999x32 CTC output tensor
    ])

    amlnn.config(
        # export_intermediate=True,
        quantized_dtype="w16a16",
        activation_dtype="f16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}",
    )

    amlnn.compile()
    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]
    output_path = output_dir / generated_path.name

    if generated_path != output_path:
        shutil.copy2(generated_path, output_path)

    if not output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {output_path}")

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
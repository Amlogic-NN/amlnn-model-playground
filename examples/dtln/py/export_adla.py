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
 
"""
Convert DTLN ONNX models (model_1 + model_2) to ADLA
"""

import argparse
import shutil
from pathlib import Path
from amlnn.api import AMLNN


def snapshot_adla_files(search_dir):
    return {p: p.stat().st_mtime for p in search_dir.rglob("*.adla")}


def find_updated_adla_files(search_dir, known_files):
    current = snapshot_adla_files(search_dir)
    updated = [
        p for p, m in current.items()
        if p not in known_files or m > known_files[p]
    ]
    return sorted(updated, key=lambda p: p.stat().st_mtime, reverse=True)


def convert_model(onnx_path, output_path, target_platform):
    print(f"\nConverting: {onnx_path}")

    search_dir = Path.cwd()
    known_files = snapshot_adla_files(search_dir)

    amlnn = AMLNN()

    amlnn.load_onnx(model=onnx_path)

    amlnn.config(
        quantized_dtype="w8a16",
        activation_dtype="f16",
        target_platform=f"PRODUCT_PID0XA{target_platform.zfill(3)}",
        export_intermediate=True,
    )

    amlnn.compile()

    amlnn.export_adla()

    new_files = find_updated_adla_files(search_dir, known_files)
    if not new_files:
        raise RuntimeError("No ADLA generated!")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if new_files[0].resolve() != output_path.resolve():
        shutil.copy2(new_files[0], output_path)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DTLN ONNX → ADLA")

    parser.add_argument("--model1", required=True, help="DTLN model_1.onnx")
    parser.add_argument("--model2", required=True, help="DTLN model_2.onnx")
    parser.add_argument("--target-platform", required=True)
    parser.add_argument("--out-dir", default="output")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # model_1
    convert_model(
        args.model1,
        out_dir / "dtln_model_1.adla",
        args.target_platform
    )

    # model_2
    convert_model(
        args.model2,
        out_dir / "dtln_model_2.adla",
        args.target_platform
    )


if __name__ == "__main__":
    main()

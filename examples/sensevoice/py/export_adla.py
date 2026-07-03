#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import shutil
import sys
from pathlib import Path

from amlnn.api import AMLNN


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
    parser = argparse.ArgumentParser(description="Export SenseVoice ONNX to ADLA")
    parser.add_argument("--onnx", required=True, help="Path to SenseVoice 3-input ONNX model")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 003, 005, 006, 007")
    parser.add_argument("--adla", default="../model", help="Output .adla path (default: ../model)")
    args = parser.parse_args()

    search_dir = Path.cwd()
    known_adla_files = snapshot_adla_files(search_dir)

    nn = AMLNN(log_level="INFO")

    print(f"[1/4] Loading ONNX model: {args.onnx}")
    ret = nn.load_onnx(
        model=args.onnx,
        inputs=["x", "language", "text_norm"],
        input_shapes=[[1, 100, 560], [1], [1]],
    )
    if ret is not None and ret != 0:
        print(f"[ERROR] load_onnx failed, return code: {ret}", file=sys.stderr)
        sys.exit(1)

    print("[2/4] Configuring conversion parameters")
    ret = nn.config(
        quantized_dtype="w8a16",
        activation_dtype="f16",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}",
    )
    if ret is not None and ret != 0:
        print(f"[ERROR] config failed, return code: {ret}", file=sys.stderr)
        sys.exit(1)

    print("[3/4] Compiling model (no calibration dataset, random calibration only)")
    ret = nn.compile()
    if ret is not None and ret != 0:
        print(f"[ERROR] compile failed, return code: {ret}", file=sys.stderr)
        sys.exit(1)

    print("[4/4] Exporting ADLA model")
    ret = nn.export_adla()
    if ret is not None and ret != 0:
        print(f"[ERROR] export_adla failed, return code: {ret}", file=sys.stderr)
        sys.exit(1)

    nn.uninit()

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
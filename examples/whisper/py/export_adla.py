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


def get_output_path(adla_arg, model_path):
    requested_path = Path(adla_arg)

    if requested_path.suffix.lower() == ".adla":
        return requested_path

    return requested_path / f"{model_path.stem}.adla"

def main():
    parser = argparse.ArgumentParser(description="Export Quantized TFLite to ADLA")
    parser.add_argument("--encoder-tflite", required=True, help="Path to Encoder TFLite model")
    parser.add_argument("--decoder-tflite", required=True, help="Path to Decoder TFLite model")
    parser.add_argument("--target-platform", required=True, help="Platform ID, e.g. 001, 002, 003")
    parser.add_argument("--adla", default="../model", help="Output .adla file or directory (default: ../model)")
    args = parser.parse_args()

    encoder_model_path = Path(args.encoder_tflite).resolve()

    if not encoder_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {encoder_model_path}")

    encoder_output_path = get_output_path(args.adla, encoder_model_path)
    encoder_output_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), encoder_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()
    amlnn.load_tflite(model=str(encoder_model_path), quantized_model=True)
    amlnn.config(
        quantized_dtype="w8a8",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
    )
    amlnn.compile()
    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("Encoder export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]

    if generated_path != encoder_output_path.resolve():
        shutil.copy2(generated_path, encoder_output_path)

    if not encoder_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {encoder_output_path}")

    print(f"saved: {encoder_output_path.resolve()}")

    decoder_model_path = Path(args.decoder_tflite).resolve()

    if not decoder_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {decoder_model_path}")

    decoder_output_path = get_output_path(args.adla, decoder_model_path)
    decoder_output_path.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = {Path.cwd().resolve(), decoder_model_path.parent}
    known_adla_files = snapshot_adla_files(search_dirs)

    amlnn = AMLNN()
    amlnn.load_tflite(model=str(decoder_model_path), quantized_model=True)
    amlnn.config(
        quantized_dtype="w8a8",
        target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}"
    )
    amlnn.compile()
    amlnn.export_adla()
    amlnn.uninit()

    updated_adla_files = find_updated_adla_files(search_dirs, known_adla_files)
    if not updated_adla_files:
        raise RuntimeError("Decoder export_adla did not create or update a .adla file")

    generated_path = updated_adla_files[0]

    if generated_path != decoder_output_path.resolve():
        shutil.copy2(generated_path, decoder_output_path)

    if not decoder_output_path.is_file():
        raise RuntimeError(f"Failed to save ADLA model: {decoder_output_path}")

    print(f"saved: {decoder_output_path.resolve()}")


if __name__ == "__main__":
    main()
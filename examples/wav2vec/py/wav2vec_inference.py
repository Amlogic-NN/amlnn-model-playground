# -*- coding: utf-8 -*-

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
import glob
import os

import librosa
import numpy as np
from amlnn.api import AMLNN

SAMPLE_RATE = 16000
TARGET_SAMPLES = 320000
OVERLAP_SECONDS = 2
OUTPUT_CHANNELS = 32

# 0-3 are special CTC tokens, and 4 is the space character.
TOKENIZER_DICT = {
    0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>", 4: "|", 5: "E", 6: "T", 7: "A", 8: "O", 9: "N", 10: "I",
    11: "H", 12: "S", 13: "R", 14: "D", 15: "L", 16: "U", 17: "M", 18: "W", 19: "C", 20: "F", 21: "G",
    22: "Y", 23: "P", 24: "B", 25: "V", 26: "K", 27: "'", 28: "X", 29: "J", 30: "Q", 31: "Z",
}


def preprocess_audio(audio_path, sample_rate, target_samples, overlap_samples):
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")

    step_samples = target_samples - 2 * overlap_samples

    if step_samples <= 0:
        raise ValueError("The left and right overlap must total less than the model input length")

    segments = []
    start = 0

    while True:
        segment = waveform[start:start + target_samples]
        real_samples = segment.size

        # Normalize only the real audio samples so zero-padding remains zero.
        mean = np.mean(segment, dtype=np.float64)
        variance = np.var(segment, dtype=np.float64)
        segment = (segment - mean) / np.sqrt(variance + 1e-7)
        segment = segment.astype(np.float32)

        if real_samples < target_samples:
            segment = np.pad(
                segment,
                (0, target_samples - real_samples),
                mode="constant",
                constant_values=0.0,
            )

        segments.append((segment, real_samples))

        if start + target_samples >= waveform.size:
            break

        start += step_samples

    return segments


def ctc_decode(predictions):
    compressed_sequence = []

    for index, token_id in enumerate(predictions):
        if index == 0 or token_id != predictions[index - 1]:
            compressed_sequence.append(int(token_id))

    transcription = []

    for token_id in compressed_sequence:
        if token_id <= 3:
            continue
        if token_id == 4:
            transcription.append(" ")
        else:
            transcription.append(TOKENIZER_DICT.get(token_id, ""))

    return "".join(transcription).strip()


def get_audio_files(audio_dir):
    audio_files = []

    for extension in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
        audio_files.extend(glob.glob(os.path.join(audio_dir, extension)))
        audio_files.extend(glob.glob(os.path.join(audio_dir, extension.upper())))

    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(description="Wav2Vec2 Speech Recognition Demo")
    parser.add_argument("--adla", required=True, help="Path to the Wav2Vec2 .adla model")
    parser.add_argument("--audio-dir", required=True, help="Directory containing test audio files")
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.adla)

    tensor_info = amlnn.get_tensor_info()

    if len(tensor_info["inputs"]) != 1:
        raise ValueError(f"Expected 1 model input, got {len(tensor_info['inputs'])}")
    if len(tensor_info["outputs"]) != 1:
        raise ValueError(f"Expected 1 model output, got {len(tensor_info['outputs'])}")

    input_info = tensor_info["inputs"][0]
    output_info = tensor_info["outputs"][0]

    input_shape = tuple(int(value) for value in input_info["dims"])
    output_shape = tuple(int(value) for value in output_info["dims"])
    input_type = int(input_info["type"])
    output_type = int(output_info["type"])

    if input_type != 0:
        raise ValueError(f"Expected FP32 model input type 0, got {input_type}")
    if output_type != 0:
        raise ValueError(f"Expected FP32 model output type 0, got {output_type}")

    target_samples = int(np.prod(input_shape))
    output_channels = output_shape[-1]
    output_steps = int(np.prod(output_shape)) // output_channels
    overlap_samples = OVERLAP_SECONDS * SAMPLE_RATE
    overlap_output_steps = int(round(overlap_samples * output_steps / target_samples))

    if target_samples != TARGET_SAMPLES:
        raise ValueError(f"Expected {TARGET_SAMPLES} input samples, got {target_samples}")
    if output_channels != OUTPUT_CHANNELS:
        raise ValueError(f"Expected {OUTPUT_CHANNELS} output channels, got {output_channels}")

    print(amlnn.get_sdk_version())
    print(f"Using ADLA model: {args.adla}")
    print(f"Input shape: {input_shape}")
    print("Input type: FP32")
    print(f"Output shape: {output_shape}")
    print("Output type: FP32")

    audio_files = get_audio_files(args.audio_dir)

    if not audio_files:
        print("No audio files found.")
        amlnn.uninit()
        return

    for file_index, audio_path in enumerate(audio_files, 1):
        print("=" * 60)
        print(f"Processing [{file_index}/{len(audio_files)}]: {os.path.basename(audio_path)}")
        print("=" * 60)

        try:
            segments = preprocess_audio(
                audio_path,
                SAMPLE_RATE,
                target_samples,
                overlap_samples,
            )

            print(f"Segments: {len(segments)}")
            retained_logits = []

            for segment_index, (waveform, real_samples) in enumerate(segments, 1):
                print(f"Processing segment [{segment_index}/{len(segments)}]...")

                # Model input is FP32 [1, 1, 1, 320000]. Do not quantize it in Python.
                input_tensor = waveform.reshape(input_shape).astype(np.float32)

                outputs = amlnn.inference(inputs=[input_tensor])
                output_tensor = np.asarray(outputs[0], dtype=np.float32)

                expected_output_elements = int(np.prod(output_shape))
                if output_tensor.size != expected_output_elements:
                    raise ValueError(
                        f"Output contains {output_tensor.size} elements, expected {expected_output_elements}"
                    )

                output_tensor = output_tensor.reshape(output_shape)
                logits = output_tensor.reshape(-1, output_channels)

                valid_output_steps = int(round(real_samples * output_steps / target_samples))
                valid_output_steps = min(max(valid_output_steps, 1), output_steps)
                keep_start = 0 if segment_index == 1 else overlap_output_steps
                keep_end = valid_output_steps if segment_index == len(segments) else output_steps - overlap_output_steps

                if keep_end <= keep_start:
                    raise ValueError(
                        f"Invalid retained output range [{keep_start}:{keep_end}] for segment {segment_index}"
                    )

                retained_logits.append(logits[keep_start:keep_end])

            combined_logits = np.concatenate(retained_logits, axis=0)
            predicted_ids = np.argmax(combined_logits, axis=-1)
            final_transcription = ctc_decode(predicted_ids)
            print(f"Transcription: {final_transcription}")

        except Exception as error:
            print(f"Error processing {os.path.basename(audio_path)}: {error}")

    print()
    print("=" * 60)
    print(amlnn.get_perf_info())
    # amlnn.perf_visualize()
    amlnn.uninit()


if __name__ == "__main__":
    main()
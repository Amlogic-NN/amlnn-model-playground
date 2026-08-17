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

import os
import glob
import argparse
import csv
import numpy as np
import librosa
from amlnn.api import AMLNN

def load_class_names(csv_path: str):
    class_names = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if len(row) >= 3:
                    class_names[int(row[0])] = row[2]
        return class_names
    except Exception as e:
        print(f"Warning: Could not load class names from '{csv_path}'. Fallback to generic IDs.")
        return {}

def preprocess_audio(audio_path: str, sr: int = 16000, max_duration: float = 15.0):
    # librosa automatically resamples the audio to 16000Hz if it isn't already
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Normalize
    max_val = np.max(np.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val

    # Truncate
    max_samples = int(max_duration * sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    window_size = 15360
    step_size = 7680

    # Pad
    if len(waveform) < window_size:
        waveform = np.pad(waveform, (0, window_size - len(waveform)), mode='constant')

    # Frame
    frames = []
    for i in range(0, len(waveform) - window_size + 1, step_size):
        frames.append(waveform[i:i + window_size])

    return frames

def prepare_tensor(frame: np.ndarray):
    input_tensor = frame.astype(np.float32).reshape(1, 1, 1, 15360)
    return input_tensor

def main():
    parser = argparse.ArgumentParser(description="YAMNet Audio Classification")
    parser.add_argument('--adla', required=True, help='Path to .adla yamnet model')
    parser.add_argument('--audio-dir', required=True, help='Directory containing test audio')
    parser.add_argument('--labels', required=True, help='Path to yamnet_class_map.csv')
    args = parser.parse_args()

    class_names = load_class_names(args.labels)

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native")
    amlnn.load_model(path=args.adla)

    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext.upper())))
    audio_files.sort()

    if not audio_files:
        print("No audio files found.")
        amlnn.uninit()
        return

    for i, audio_path in enumerate(audio_files, 1):
        print(f"{'-'*60}\nProcessing [{i}/{len(audio_files)}]: {os.path.basename(audio_path)}")

        try:
            frames = preprocess_audio(audio_path)
            file_predictions = []

            for frame in frames:
                input_tensor = prepare_tensor(frame)
                outputs = amlnn.inference(inputs=[input_tensor])
                preds = outputs[0].flatten()
                file_predictions.append(preds)

            if file_predictions:
                mean_scores = np.mean(file_predictions, axis=0) 
                top_5_indices = np.argsort(mean_scores)[::-1][:5]

                for rank, class_idx in enumerate(top_5_indices, 1):
                    class_name = class_names.get(class_idx, f"Class_{class_idx}")
                    confidence = mean_scores[class_idx]
                    print(f"  {rank}. {class_name:<30} ({confidence:.4f})")

        except Exception as e:
            print(f"Error processing {os.path.basename(audio_path)}: {e}")

    amlnn.uninit()

if __name__ == "__main__":
    main()
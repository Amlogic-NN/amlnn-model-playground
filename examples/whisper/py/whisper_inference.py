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
import os

from transformers import WhisperProcessor
import librosa
import numpy as np
from amlnn.api import AMLNN

SAMPLE_RATE = 16000
TARGET_SAMPLES = 30 * SAMPLE_RATE
OVERLAP_SECONDS = 2
LANGUAGE = "en"


def get_audio_segments(audio_path, sample_rate, target_samples, overlap_samples):
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")

    step_samples = target_samples - overlap_samples
    if step_samples <= 0:
        raise ValueError("Overlap must be shorter than the model input length")

    segments = []
    start = 0

    while True:
        segment = waveform[start:start + target_samples]
        segments.append(segment)

        if start + target_samples >= waveform.size:
            break

        start += step_samples

    return segments


def preprocess_audio(waveform, processor, input_shape, s, zp, tensor_type):
    input_features = processor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np"
    ).input_features
    input_features = np.asarray(input_features, dtype=np.float32)

    expected_elements = int(np.prod(input_shape))
    if input_features.size != expected_elements:
        raise ValueError(
            f"input_features contains {input_features.size} elements, "
            f"expected {expected_elements} for input shape {input_shape}"
        )

    input_features = input_features.reshape(input_shape)

    if tensor_type == 0:  # FP32 & FP16
        input_tensor = input_features.astype(np.float32)
    elif tensor_type in (2, 3, 4):
        raw_val = np.round(input_features / s + zp)

        if tensor_type == 2:    # Int8
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:  # Uint8
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:                   # Int16
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support encoder input tensor type: {tensor_type}")

    return np.ascontiguousarray(input_tensor)


def prepare_encoder_hidden_states(encoder_output, input_shape, s, zp, tensor_type):
    encoder_output = np.asarray(encoder_output, dtype=np.float32)

    expected_elements = int(np.prod(input_shape))
    if encoder_output.size != expected_elements:
        raise ValueError(
            f"Encoder output contains {encoder_output.size} elements, "
            f"decoder encoder_hidden_states expects {expected_elements}"
        )

    encoder_output = encoder_output.reshape(input_shape)

    if tensor_type == 0:  # FP32 & FP16
        input_tensor = encoder_output.astype(np.float32)
    elif tensor_type in (2, 3, 4):
        raw_val = np.round(encoder_output / s + zp)

        if tensor_type == 2:    # Int8
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:  # Uint8
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:                   # Int16
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(
            f"Does not support decoder encoder_hidden_states tensor type: {tensor_type}"
        )

    return np.ascontiguousarray(input_tensor)


def get_decoder_tokens(processor):
    tokenizer = processor.tokenizer
    tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task="transcribe",
        predict_timestamps=False
    )

    decoder_tokens = list(tokenizer.prefix_tokens)
    if not decoder_tokens:
        raise ValueError("Whisper tokenizer returned no decoder prefix tokens")

    if tokenizer.eos_token_id is None:
        raise ValueError("Whisper tokenizer does not define an end token")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    return decoder_tokens, int(tokenizer.eos_token_id), int(pad_token_id)


def run_decoder_loop(
    decoder_amlnn,
    decoder_ids_shape,
    decoder_hidden_states,
    decoder_output_shape,
    processor
):
    decoder_length = int(np.prod(decoder_ids_shape))
    vocab_size = int(decoder_output_shape[-1])
    output_steps = int(np.prod(decoder_output_shape)) // vocab_size

    if output_steps != decoder_length:
        raise ValueError(
            f"Decoder output has {output_steps} positions, "
            f"but decoder input length is {decoder_length}"
        )

    decoder_tokens, token_eot, pad_token_id = get_decoder_tokens(processor)

    if len(decoder_tokens) >= decoder_length:
        raise ValueError(
            f"Decoder prefix has {len(decoder_tokens)} tokens, "
            f"but decoder input length is {decoder_length}"
        )

    while len(decoder_tokens) < decoder_length:
        decoder_input_ids = np.full(decoder_length, pad_token_id, dtype=np.int64)
        decoder_input_ids[:len(decoder_tokens)] = decoder_tokens
        decoder_input_ids = decoder_input_ids.reshape(decoder_ids_shape)

        outputs = decoder_amlnn.inference(
            inputs=[decoder_input_ids, decoder_hidden_states]
        )

        if outputs is None or len(outputs) == 0:
            raise ValueError("Decoder inference returned no outputs")

        logits = np.asarray(outputs[0], dtype=np.float32)

        expected_elements = int(np.prod(decoder_output_shape))
        if logits.size != expected_elements:
            raise ValueError(
                f"Decoder output contains {logits.size} elements, "
                f"expected {expected_elements}"
            )

        logits = logits.reshape(output_steps, vocab_size)
        next_token = int(np.argmax(logits[len(decoder_tokens) - 1]))
        decoder_tokens.append(next_token)

        if next_token == token_eot:
            break

    return processor.decode(
        decoder_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()


def normalize_word(word):
    return "".join(character.lower() for character in word if character.isalnum() or character == "'")


def merge_transcriptions(transcriptions):
    combined = ""

    for transcription in transcriptions:
        transcription = transcription.strip()
        if not transcription:
            continue

        if not combined:
            combined = transcription
            continue

        previous_words = combined.split()
        current_words = transcription.split()
        max_overlap = min(len(previous_words), len(current_words))
        matched_words = 0

        for count in range(max_overlap, 0, -1):
            previous_suffix = [normalize_word(word) for word in previous_words[-count:]]
            current_prefix = [normalize_word(word) for word in current_words[:count]]

            if previous_suffix == current_prefix and all(previous_suffix):
                matched_words = count
                break

        if matched_words > 0:
            combined = " ".join(previous_words + current_words[matched_words:])
            continue

        previous_last = normalize_word(previous_words[-1])
        current_first = normalize_word(current_words[0])

        if len(previous_last) >= 3 and current_first.startswith(previous_last):
            combined = " ".join(previous_words[:-1] + current_words)
        elif len(current_first) >= 3 and previous_last.startswith(current_first):
            combined = " ".join(previous_words + current_words[1:])
        else:
            combined = combined.rstrip() + " " + transcription.lstrip()

    return combined.strip()


def main():
    parser = argparse.ArgumentParser(description="Whisper ADLA Demo")
    parser.add_argument("--enc", required=True, help="Path to encoder .adla model")
    parser.add_argument("--dec", required=True, help="Path to decoder .adla model")
    parser.add_argument("--tokenizer", required=True, help="Path to local Whisper processor directory")
    parser.add_argument("--audio-file", required=True, help="Path to input audio file")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"Audio file not found: {args.audio_file}")
        return 0

    processor = WhisperProcessor.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        clean_up_tokenization_spaces=False
    )

    encoder_amlnn = AMLNN()
    decoder_amlnn = AMLNN()

    encoder_amlnn.init_runtime(mode="native", enable_perf=True)
    encoder_amlnn.load_model(path=args.enc)
    encoder_tensor_info = encoder_amlnn.get_tensor_info()

    decoder_amlnn.init_runtime(mode="native", enable_perf=True)
    decoder_amlnn.load_model(path=args.dec)
    decoder_tensor_info = decoder_amlnn.get_tensor_info()

    print(encoder_amlnn.get_sdk_version())

    encoder_input_attr = encoder_tensor_info["inputs"][0]
    encoder_output_attr = encoder_tensor_info["outputs"][0]

    decoder_ids_attr = decoder_tensor_info["inputs"][0]
    decoder_hidden_attr = decoder_tensor_info["inputs"][1]
    decoder_output_attr = decoder_tensor_info["outputs"][0]

    encoder_input_shape = tuple(int(value) for value in encoder_input_attr["dims"])
    encoder_output_shape = tuple(int(value) for value in encoder_output_attr["dims"])
    decoder_ids_shape = tuple(int(value) for value in decoder_ids_attr["dims"])
    decoder_hidden_shape = tuple(int(value) for value in decoder_hidden_attr["dims"])
    decoder_output_shape = tuple(int(value) for value in decoder_output_attr["dims"])

    encoder_s = float(encoder_input_attr["scale"])
    encoder_zp = int(encoder_input_attr["zp"])
    encoder_type = int(encoder_input_attr["type"])

    decoder_hidden_s = float(decoder_hidden_attr["scale"])
    decoder_hidden_zp = int(decoder_hidden_attr["zp"])
    decoder_hidden_type = int(decoder_hidden_attr["type"])

    print(f"Encoder input: name={encoder_input_attr['name']}, shape={encoder_input_shape}")
    print(f"Encoder output: name={encoder_output_attr['name']}, shape={encoder_output_shape}")
    print(f"Decoder input 0: name={decoder_ids_attr['name']}, shape={decoder_ids_shape}")
    print(f"Decoder input 1: name={decoder_hidden_attr['name']}, shape={decoder_hidden_shape}")
    print(f"Decoder output: name={decoder_output_attr['name']}, shape={decoder_output_shape}")

    print("=" * 60)
    print(f"Processing audio: {os.path.basename(args.audio_file)}")
    print("=" * 60)

    try:
        overlap_samples = OVERLAP_SECONDS * SAMPLE_RATE
        segments = get_audio_segments(
            args.audio_file,
            SAMPLE_RATE,
            TARGET_SAMPLES,
            overlap_samples
        )

        print(f"Segments: {len(segments)}")
        segment_transcriptions = []

        for segment_index, waveform in enumerate(segments, 1):
            print(f"Processing segment [{segment_index}/{len(segments)}]...")

            input_features = preprocess_audio(
                waveform,
                processor,
                encoder_input_shape,
                encoder_s,
                encoder_zp,
                encoder_type
            )

            encoder_outputs = encoder_amlnn.inference(
                inputs=[input_features]
            )

            if encoder_outputs is None or len(encoder_outputs) == 0:
                raise ValueError("Encoder inference returned no outputs")

            encoder_output = np.asarray(encoder_outputs[0], dtype=np.float32)

            expected_encoder_elements = int(np.prod(encoder_output_shape))
            if encoder_output.size != expected_encoder_elements:
                raise ValueError(
                    f"Encoder output contains {encoder_output.size} elements, "
                    f"expected {expected_encoder_elements}"
                )

            decoder_hidden_states = prepare_encoder_hidden_states(
                encoder_output,
                decoder_hidden_shape,
                decoder_hidden_s,
                decoder_hidden_zp,
                decoder_hidden_type
            )

            transcription = run_decoder_loop(
                decoder_amlnn,
                decoder_ids_shape,
                decoder_hidden_states,
                decoder_output_shape,
                processor
            )

            segment_transcriptions.append(transcription)

        final_transcription = merge_transcriptions(segment_transcriptions)
        print(f"Transcription: {final_transcription}")

    except Exception as e:
        print(f"Error processing {os.path.basename(args.audio_file)}: {e}")

    print("=" * 60)
    print("Encoder performance:")
    print(encoder_amlnn.get_perf_info())
    print("Decoder performance:")
    print(decoder_amlnn.get_perf_info())

    # encoder_amlnn.perf_visualize()
    # decoder_amlnn.perf_visualize()

    encoder_amlnn.uninit()
    decoder_amlnn.uninit()

    return 0


if __name__ == "__main__":
    main()
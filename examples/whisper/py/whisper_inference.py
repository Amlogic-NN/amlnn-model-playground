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

# This inference script is designed for whisper-base-en or whisper-tiny-en models using AMLNN.

import argparse
import os
import numpy as np
from transformers import WhisperProcessor
import librosa
from amlnn.api import AMLNN

# Special tokens
TOKEN_EOT = 50256  # End of text token
TOKEN_SOT = 50257  # Start of transcript token
TOKEN_NOT = 50362  # No speech token (for tiny_en/base_en)

def preprocess_audio(file_path, processor, max_length=3000, tensor_type=0):
    """
    Load and preprocess audio file to mel spectrogram features.

    Args:
        file_path: Path to audio file
        processor: WhisperProcessor instance
        max_length: Maximum length of mel spectrogram frames (default: 3000)

    Returns:
        input_features: Preprocessed mel spectrogram features with shape (1, 80, max_length)
    """
    # Load audio file
    audio_signal, sampling_rate = librosa.load(file_path, sr=16000)

    # Process audio using WhisperProcessor
    input_features = processor(audio_signal, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Convert from PyTorch tensor to numpy array
    if hasattr(input_features, 'numpy'):
        input_features = input_features.numpy()
    else:
        input_features = np.array(input_features)

    # Truncate or pad to max_length
    if input_features.shape[2] > max_length:
        input_features = input_features[:, :, :max_length]
    elif input_features.shape[2] < max_length:
        # Pad with zeros
        pad_length = max_length - input_features.shape[2]
        input_features = np.pad(input_features, ((0, 0), (0, 0), (0, pad_length)), mode='constant')

    # AMLNN requires 1D or 4D input. Whisper mel is (1, 80, T).
    # Convert to NHWC: (1, 1, 80, T)
    if input_features.ndim != 3 or input_features.shape[0] != 1 or input_features.shape[1] != 80:
        raise ValueError(f"[Error] Unexpected input_features shape: {input_features.shape}, expected (1, 1, 80, T)")

    input_features = input_features[:, None, :, :]  # (1, 1, 80, T)

    print(f"[Info] input_features shape: {input_features.shape}")
    if tensor_type == 0:
        input_features = input_features.astype(np.float32)
    elif tensor_type == 1:
        input_features = input_features.astype(np.float16)
    # input_features = (input_features / 0.007843 - 29).astype(np.int8)

    return input_features

def run_encoder(encoder_amlnn, input_features):
    """
    Run encoder inference.

    Args:
        encoder_amlnn: AMLNN instance for encoder
        input_features: Preprocessed mel spectrogram features

    Returns:
        encoder_outputs: Encoder output features
    """
    # Run encoder inference
    outputs = encoder_amlnn.inference(
        inputs=[input_features],
        inputs_data_format='NHWC',
        outputs_data_format='NHWC'
    )

    if outputs is None or len(outputs) == 0:
        raise ValueError("[Error] Encoder inference returned None or empty outputs")

    encoder_outputs = outputs[0]
    print(f"[Info] encoder_outputs shape: {encoder_outputs.shape}")

    return encoder_outputs

def run_decoder_loop(decoder_amlnn, encoder_outputs, processor, max_new_tokens=45, stream_print=True, tensor_type=0):
    """
    Run decoder inference loop to generate transcription tokens.

    Args:
        decoder_amlnn: AMLNN instance for decoder
        encoder_outputs: Encoder output features
        processor: WhisperProcessor instance for decoding
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        transcription: Decoded transcription text
    """
    # Initialize decoder input with start tokens [50257, 50362] for tiny_en/base_en
    decoder_input_ids = np.array([[TOKEN_SOT, TOKEN_NOT]], dtype=np.int64)

    # Decoder input shape is typically (1, 48) - pad to this length
    decoder_input_length = 48
    vocab_size = 51864  # Whisper vocab size

    # Track current sequence length (starts at 2 for initial tokens)
    current_seq_len = 2

    # For streaming print (print only newly generated text)
    prev_text = ""

    # Decoder inference loop
    for step in range(max_new_tokens):
        # Prepare decoder input tensor - pad to decoder_input_length
        input_ids_length = decoder_input_ids.shape[1]
        # AMLNN requires 1D or 4D input. Token ids are naturally 2D (1, L),
        # so we expand to 4D as you requested: (1, 1, 1, L)
        decoder_input_padded_2d = np.zeros((1, decoder_input_length), dtype=np.int64)
        decoder_input_padded_2d[:, :input_ids_length] = decoder_input_ids
        decoder_input_padded = decoder_input_padded_2d[:, None, None, :]  # (1, 1, 1, L)

        # Prepare inputs: [decoder_input_ids, encoder_outputs]
        # The decoder expects two inputs: encoder outputs and decoder input ids
        if tensor_type == 1:
            encoder_outputs = encoder_outputs.astype(np.float16)
        decoder_inputs = [decoder_input_padded, encoder_outputs]

        # Run decoder inference
        outputs = decoder_amlnn.inference(
            inputs=decoder_inputs,
            inputs_data_format='NHWC',
            outputs_data_format='NHWC'
        )

        if outputs is None or len(outputs) == 0:
            raise ValueError("[Error] Decoder inference returned None or empty outputs")

        # Get logits from decoder output
        # Output shape is typically (1, 1, max_seq_len, vocab_size) = (1, 1, 64, 51864)
        logits = outputs[0]  # Shape: (1, 1, 64, 51864)

        # Extract logits for the current position
        # According to C++ code: begin_count = (id_shape - 1) * 51864
        # We need logits at position (current_seq_len - 1)
        if len(logits.shape) == 4:
            # logits shape: (1, 1, 64, 51864)
            # Extract logits at position (current_seq_len - 1)
            pos_idx = current_seq_len - 1
            if pos_idx < logits.shape[2]:
                next_token_logits = logits[0, 0, pos_idx, :]  # Shape: (51864,)
            else:
                # Fallback to last position if index out of range
                next_token_logits = logits[0, -1, :]
        else:
            # Handle unexpected shape
            next_token_logits = logits.flatten()[:vocab_size]

        # Get next token (argmax)
        next_token = int(np.argmax(next_token_logits))

        # Update decoder input ids
        decoder_input_ids = np.concatenate((decoder_input_ids, np.array([[next_token]])), axis=1)
        current_seq_len += 1

        # Stream printing: decode current sequence and print delta only
        if stream_print:
            try:
                cur_text = processor.decode(decoder_input_ids[0], skip_special_tokens=True)
                if cur_text.startswith(prev_text):
                    delta = cur_text[len(prev_text):]
                else:
                    # Fallback: tokenizer may change previous bytes; just print full line
                    delta = cur_text
                    prev_text = ""
                if delta:
                    print(delta, end="", flush=True)
                prev_text = cur_text
            except Exception:
                # If decoding fails for some reason, don't break inference
                pass

        # Stop if end token is generated
        if next_token == TOKEN_EOT:
            print("\n[Info] End token generated, stopping generation.")
            break

        # Safety check: prevent exceeding maximum decoder input length
        if current_seq_len >= decoder_input_length:
            print("\n[Info] Reached maximum decoder input length, stopping generation.")
            break

    # Decode transcription using processor
    transcription = processor.batch_decode(decoder_input_ids, skip_special_tokens=True)

    # If streaming printed without newline, end the line for clean output
    if stream_print:
        print()

    return transcription

def main():
    parser = argparse.ArgumentParser(description="Whisper Tiny EN Inference using AMLNN")

    parser.add_argument('--encoder-model-path', required=True, help='Path to encoder model')
    parser.add_argument('--decoder-model-path', required=True, help='Path to decoder model')
    parser.add_argument('--tokenizer-dir', required=True, help='Path to Whisper tokenizer directory (for processor)')
    parser.add_argument('--audio-file', default=None, help='Path to input audio file (.wav) - optional, will prompt if not provided')
    parser.add_argument('--max-new-tokens', type=int, default=45, help='Maximum number of tokens to generate')

    args = parser.parse_args()

    # Validate inputs
    # if not os.path.exists(args.encoder_model_path):
    #     print(f"[Error] Encoder model not found: {args.encoder_model_path}")
    #     return -1

    # if not os.path.exists(args.decoder_model_path):
    #     print(f"[Error] Decoder model not found: {args.decoder_model_path}")
    #     return -1

    # Load processor
    print(f"[Info] Loading WhisperProcessor from: {args.tokenizer_dir}")
    processor = WhisperProcessor.from_pretrained(args.tokenizer_dir)

    # Initialize encoder
    # print(f"[Info] Initializing encoder model: {args.encoder_model_path}")
    encoder_amlnn = AMLNN()

    encoder_amlnn.init_runtime(mode="native", enable_perf=True)

    encoder_amlnn.load_model(path=args.encoder_model_path)

    encoder_tensor_info = encoder_amlnn.get_tensor_info()

    # Initialize decoder
    # print(f"[Info] Initializing decoder model: {args.decoder_model_path}")
    decoder_amlnn = AMLNN()

    decoder_amlnn.init_runtime(mode="native", enable_perf=True)

    decoder_amlnn.load_model(path=args.decoder_model_path)

    decoder_tensor_info = decoder_amlnn.get_tensor_info()


    try:
        # Interactive loop: process audio files one by one
        while True:
            # Get audio file path from user input
            if args.audio_file:
                # Use provided audio file (first iteration only)
                audio_file = args.audio_file
                args.audio_file = None  # Clear for next iteration
            else:
                # Prompt user for audio file path
                print("\n" + "=" * 60)
                print("[Info] Audio Path:")
                audio_file = input().strip()

            # Check for exit command
            if audio_file.lower() == "exit":
                print("\n[Info] Exiting...")
                break

            # Validate input
            if not audio_file:
                print("\n[Debug] Please enter wav path")
                continue

            if len(audio_file) < 4 or not audio_file.lower().endswith('.wav'):
                print("\n[Error] Invalid wav path or file does not exist, please try again")
                continue

            if not os.path.exists(audio_file):
                print(f"\n[Error] Audio file not found: {audio_file}")
                continue

            try:
                # Preprocess audio
                encoder_tensor_attr = encoder_tensor_info["inputs"][0]
                encoder_tensor_type = int(encoder_tensor_attr["type"])
                print(f"\n[Info] Preprocessing audio: {audio_file}")
                input_features = preprocess_audio(audio_file, processor, tensor_type=encoder_tensor_type)

                # Run encoder
                print("\n[Info] Running encoder inference...")
                encoder_outputs = run_encoder(encoder_amlnn, input_features)

                # Run decoder loop
                decoder_tensor_attr = decoder_tensor_info["inputs"][1]
                decoder_tensor_type = int(decoder_tensor_attr["type"])
                print("\n[Info] Running decoder inference loop...")
                print("[Info] Audio Text:")
                transcription = run_decoder_loop(decoder_amlnn, encoder_outputs, processor, args.max_new_tokens, stream_print=True, tensor_type=decoder_tensor_type)

                # Output the transcription (already printed during streaming, but print final result)
                print("\n" + "=" * 60)
                print("[Info] Final Transcription:")
                print("=" * 60)
                if isinstance(transcription, list):
                    print(transcription[0] if len(transcription) > 0 else "[Info] No transcription generated")
                else:
                    print(transcription)
                print("=" * 60)

            except Exception as e:
                print(f"\n[Error] Error during inference: {e}")
                import traceback
                traceback.print_exc()
                print("[Info] Please try another audio file or type 'exit' to quit.")
                continue

    except KeyboardInterrupt:
        print("\n\n[Info] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n[Error] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return -1

    finally:
        # Cleanup
        encoder_amlnn.uninit()
        decoder_amlnn.uninit()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

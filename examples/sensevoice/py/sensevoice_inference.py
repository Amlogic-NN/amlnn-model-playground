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
import sys
import wave

import numpy as np

try:
    import kaldi_native_fbank as knf
except ImportError as exc:
    raise ImportError(
        "kaldi_native_fbank is required. Install with: pip install kaldi-native-fbank"
    ) from exc

from amlnn.api import AMLNN

SAMPLE_RATE = 16000
FEATURE_DIM = 80
LFR_WINDOW_SIZE = 7
LFR_WINDOW_SHIFT = 6
LFR_OUT_DIM = 560
FIXED_FRAMES = 100
VOCAB_SIZE = 25055
BLANK_ID = 0
META_TOKEN_COUNT = 4
WITH_ITN_ID = 14
WITHOUT_ITN_ID = 15

LANG_TO_ID = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
}


def parse_args():
    parser = argparse.ArgumentParser(description="SenseVoice ADLA inference demo")
    parser.add_argument("--model", required=True, help="Path to .adla model")
    parser.add_argument("--tokens", default="model/tokens.txt", help="Path to tokens.txt")
    parser.add_argument("--wav", required=True, help="Input wav file (16kHz PCM16)")
    parser.add_argument(
        "--lang", default="auto",
        choices=["auto", "zh", "en", "ja", "ko", "yue"],
        help="Language hint (default: auto)",
    )
    parser.add_argument(
        "--itn", type=int, default=0, choices=[0, 1],
        help="Enable inverse text normalization (default: 0)",
    )
    parser.add_argument(
        "--fixed-frames", type=int, default=FIXED_FRAMES,
        help=f"Fixed feature frames, must match ADLA model (default: {FIXED_FRAMES})",
    )
    parser.add_argument(
        "--mode", default="native", choices=["nnserver", "native"],
        help="Runtime mode (default: native)",
    )
    parser.add_argument(
        "--board-work-path", default="/data/nn",
        help="Working directory on board (default: /data/nn)",
    )
    parser.add_argument(
        "--input-order", default="onnx", choices=["onnx", "cpp"],
        help="Input list order: onnx=[x,language,text_norm], cpp=[text_norm,language,x]",
    )
    parser.add_argument(
        "--enable-perf", action="store_true",
        help="Print runtime performance info after inference",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print feature/logits/decode diagnostics",
    )
    return parser.parse_args()


def language_to_id(language):
    return LANG_TO_ID.get(language, 0)


def normalize_bpe_symbol(sym):
    if sym.startswith("\u2581"):
        return " " + sym[1:]
    return sym


def load_symbol_table(path):
    id2sym = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sym = normalize_bpe_symbol(parts[0])
            id2sym[int(parts[1])] = sym
    return id2sym


def read_wav_pcm16(path):
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("only PCM16 wav is supported")
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if sample_rate != SAMPLE_RATE:
        print(f"[Warning] wav sample rate is {sample_rate}, expected {SAMPLE_RATE}")

    return pcm.astype(np.float32) / 32768.0


def apply_lfr(frames):
    num_frames = frames.shape[0]
    if num_frames < LFR_WINDOW_SIZE:
        return np.zeros((0, LFR_OUT_DIM), dtype=np.float32)

    out_num_frames = (num_frames - LFR_WINDOW_SIZE) // LFR_WINDOW_SHIFT + 1
    out = np.zeros((out_num_frames, LFR_OUT_DIM), dtype=np.float32)
    for i in range(out_num_frames):
        start = i * LFR_WINDOW_SHIFT
        out[i] = frames[start:start + LFR_WINDOW_SIZE].reshape(-1)
    return out


def extract_features(samples, fixed_frames):
    if samples.size == 0:
        return np.zeros((fixed_frames, LFR_OUT_DIM), dtype=np.float32)

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.window_type = "hamming"
    opts.mel_opts.num_bins = FEATURE_DIM
    opts.mel_opts.high_freq = 0.0
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.is_librosa = False

    scaled = samples * 32768.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, scaled.tolist())
    fbank.input_finished()

    num_frames = fbank.num_frames_ready
    if num_frames <= 0:
        return np.zeros((fixed_frames, LFR_OUT_DIM), dtype=np.float32)

    frames = np.stack([fbank.get_frame(i) for i in range(num_frames)], axis=0)
    lfr = apply_lfr(frames)
    if lfr.size == 0:
        return np.zeros((fixed_frames, LFR_OUT_DIM), dtype=np.float32)

    out = np.zeros((fixed_frames, LFR_OUT_DIM), dtype=np.float32)
    copy_frames = min(lfr.shape[0], fixed_frames)
    out[:copy_frames] = lfr[:copy_frames]
    return out


def build_model_inputs(features, language_id, text_norm_id, input_order):
    x = features.astype(np.float32).reshape(1, 1, features.shape[0], LFR_OUT_DIM)
    language = np.array([[[[language_id]]]], dtype=np.int32)
    text_norm = np.array([[[[text_norm_id]]]], dtype=np.int32)
    if input_order == "cpp":
        return [text_norm, language, x]
    return [x, language, text_norm]


def ctc_greedy_decode(logits, vocab_size):
    if logits.ndim != 2:
        logits = logits.reshape(-1, vocab_size)

    tokens = []
    prev_id = -1
    for row in logits:
        best_id = int(np.argmax(row))
        if best_id != BLANK_ID and best_id != prev_id:
            tokens.append(best_id)
        prev_id = best_id
    return tokens


def split_tokens(token_ids, id2sym):
    def lookup(token_id):
        return id2sym.get(token_id, "")

    return {
        "language": lookup(token_ids[0]) if len(token_ids) >= 1 else "",
        "emotion":  lookup(token_ids[1]) if len(token_ids) >= 2 else "",
        "event":    lookup(token_ids[2]) if len(token_ids) >= 3 else "",
        "itn":      lookup(token_ids[3]) if len(token_ids) >= 4 else "",
        "text":     "".join(lookup(t) for t in token_ids[META_TOKEN_COUNT:]),
    }


def print_debug_info(features, language_id, text_norm_id, logits, token_ids):
    logits_2d = logits.reshape(-1, VOCAB_SIZE)
    pred_ids = np.argmax(logits_2d, axis=1)
    blank_ratio = float(np.mean(pred_ids == BLANK_ID))

    print(f"      [debug] feature stats: "
          f"min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    print(f"      [debug] language_id={language_id}, text_norm_id={text_norm_id}")
    print(f"      [debug] logits stats: "
          f"min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
    print(f"      [debug] blank ratio: {blank_ratio:.2%}, decoded tokens: {len(token_ids)}")
    if token_ids:
        print(f"      [debug] token ids: {token_ids[:20]}")


def recognize(amlnn, wav_path, tokens_path, language, use_itn,
              fixed_frames, input_order, debug):
    id2sym = load_symbol_table(tokens_path)
    samples = read_wav_pcm16(wav_path)
    features = extract_features(samples, fixed_frames)

    language_id = language_to_id(language)
    text_norm_id = WITH_ITN_ID if use_itn else WITHOUT_ITN_ID
    inputs = build_model_inputs(features, language_id, text_norm_id, input_order)

    print(f"      input order: {input_order}")
    print(f"      input shapes: {[arr.shape for arr in inputs]}")
    print(f"      input dtypes: {[arr.dtype for arr in inputs]}")

    outputs = amlnn.inference(
        inputs=inputs,
        inputs_data_format="NHWC",
        outputs_data_format="NHWC",
    )
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("model inference returned empty outputs")

    logits = np.asarray(outputs[0], dtype=np.float32)
    print(f"      output shape: {logits.shape}")
    token_ids = ctc_greedy_decode(logits, VOCAB_SIZE)
    if debug:
        print_debug_info(features, language_id, text_norm_id, logits, token_ids)
    return split_tokens(token_ids, id2sym)


def check_paths(paths):
    for path, label in paths:
        if not os.path.exists(path):
            print(f"[Error] {label} not found: {path}")
            sys.exit(1)


def main():
    args = parse_args()

    check_paths([
        (args.model, "model"),
        (args.tokens, "tokens"),
        (args.wav, "wav"),
    ])

    nn = AMLNN()

    print(f"[1/3] Initializing runtime (mode={args.mode})")
    nn.init_runtime(
        mode=args.mode,
        board_work_path=args.board_work_path,
        enable_perf=args.enable_perf,
    )
    print(f"      SDK version: {nn.get_sdk_version()}")

    print(f"[2/3] Loading ADLA model: {args.model}")
    nn.load_model(args.model)
    tensor_info = nn.get_tensor_info()
    print(f"      inputs:  {[t.get('name') for t in tensor_info.get('inputs', [])]}")
    print(f"      outputs: {[t.get('name') for t in tensor_info.get('outputs', [])]}")

    print(f"[3/3] Recognizing: {args.wav}")
    try:
        result = recognize(
            nn,
            args.wav,
            args.tokens,
            args.lang,
            use_itn=bool(args.itn),
            fixed_frames=args.fixed_frames,
            input_order=args.input_order,
            debug=args.debug,
        )
    finally:
        if args.enable_perf:
            print(nn.get_perf_info())
        nn.uninit()

    print()
    print(f"language: {result['language']}")
    print(f"emotion:  {result['emotion']}")
    print(f"event:    {result['event']}")
    print(f"itn:      {result['itn']}")
    print(f"text:     {result['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
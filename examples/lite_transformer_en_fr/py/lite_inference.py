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
from pathlib import Path

import numpy as np
from amlnn.api import AMLNN

MAX_TEXT_LENGTH = 64
EXPECTED_VOCAB_SIZE = 44512

BOS_TOKEN = "<s>"
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


def load_dictionary(dict_path):
    symbols = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]
    indices = {symbol: index for index, symbol in enumerate(symbols)}

    with open(dict_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip()
            if not line:
                continue

            token = line.rsplit(" ", 1)[0]
            if token not in indices:
                indices[token] = len(symbols)
                symbols.append(token)

    return symbols, indices


def load_bpe(bpe_codes_path):
    try:
        from subword_nmt.apply_bpe import BPE
    except ImportError as error:
        raise ImportError("subword-nmt is required") from error

    with open(bpe_codes_path, "r", encoding="utf-8") as file:
        return BPE(file)


def load_moses_tokenizer():
    try:
        from sacremoses import MosesTokenizer
        return MosesTokenizer(lang="en")
    except ImportError as error:
        raise ImportError("sacremoses is required") from error


def load_moses_detokenizer():
    try:
        from sacremoses import MosesDetokenizer
        return MosesDetokenizer(lang="fr")
    except ImportError as error:
        raise ImportError("sacremoses is required") from error


def preprocess_text(text, source_indices, max_len, bpe, tokenizer):
    pad_id = source_indices[PAD_TOKEN]
    eos_id = source_indices[EOS_TOKEN]
    unk_id = source_indices[UNK_TOKEN]

    tokenized_text = tokenizer.tokenize(text, return_str=True)
    bpe_text = bpe.process_line(tokenized_text)

    tokens = bpe_text.split()
    token_ids = [source_indices.get(token, unk_id) for token in tokens]

    if len(token_ids) >= max_len:
        token_ids = token_ids[:max_len - 1]

    token_ids.append(eos_id)

    input_tensor = np.full((1, 1, 1, max_len), pad_id, dtype=np.int64)
    input_tensor[0, 0, 0, :len(token_ids)] = token_ids

    return input_tensor, tokenized_text, bpe_text, token_ids


def decode_tokens(token_ids, target_symbols, detokenizer):
    tokens = []

    for token_id in token_ids:
        token = target_symbols[token_id] if 0 <= token_id < len(target_symbols) else UNK_TOKEN
        if token in (BOS_TOKEN, PAD_TOKEN, EOS_TOKEN):
            continue
        tokens.append(token)

    bpe_text = " ".join(tokens)
    decoded_text = bpe_text.replace("@@ ", "").replace("@@", "")
    decoded_text = detokenizer.detokenize(decoded_text.split())

    return bpe_text, decoded_text


def translate(amlnn, src_tokens, target_length, target_vocab_size, target_pad_id, target_eos_id):
    prev_output_tokens = np.full((1, 1, 1, target_length), target_pad_id, dtype=np.int64)
    prev_output_tokens[0, 0, 0, 0] = target_eos_id

    generated_ids = []

    for step in range(target_length):
        outputs = amlnn.inference(
            inputs=[src_tokens, prev_output_tokens]
        )

        logits = np.asarray(outputs[0], dtype=np.float32).reshape(1, target_length, target_vocab_size)
        next_token_id = int(np.argmax(logits[0, step]))

        if next_token_id == target_eos_id:
            break

        generated_ids.append(next_token_id)

        if step + 1 < target_length:
            prev_output_tokens[0, 0, 0, step + 1] = next_token_id

    return generated_ids


def main():
    parser = argparse.ArgumentParser(
        description="Lite Transformer English-to-French Translation Demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to Lite Transformer .adla model")
    parser.add_argument(
        "--assets-dir",
        required=True,
        help=(
            "Directory containing:\n"
            "  dict.en.txt\n"
            "  dict.fr.txt\n"
            "  bpecodes"
        ),
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        required=True,
        help=(
            "English sentences to translate.\n"
            "Wrap each complete sentence in quotes.\n"
            "Example:\n"
            '  --texts "Hello world." "This is a translation test."'
        ),
    )
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).resolve()
    source_dict_path = assets_dir / "dict.en.txt"
    target_dict_path = assets_dir / "dict.fr.txt"
    bpe_codes_path = assets_dir / "bpecodes"

    if not source_dict_path.is_file():
        raise FileNotFoundError(f"Missing source dictionary: {source_dict_path}")
    if not target_dict_path.is_file():
        raise FileNotFoundError(f"Missing target dictionary: {target_dict_path}")
    if not bpe_codes_path.is_file():
        raise FileNotFoundError(f"Missing BPE codes: {bpe_codes_path}")

    source_symbols, source_indices = load_dictionary(source_dict_path)
    target_symbols, target_indices = load_dictionary(target_dict_path)

    print(f"Source vocabulary size: {len(source_symbols)}")
    print(f"Target vocabulary size: {len(target_symbols)}")

    if len(source_symbols) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected source vocabulary size {EXPECTED_VOCAB_SIZE}, got {len(source_symbols)}")
    if len(target_symbols) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected target vocabulary size {EXPECTED_VOCAB_SIZE}, got {len(target_symbols)}")

    bpe = load_bpe(bpe_codes_path)
    tokenizer = load_moses_tokenizer()
    detokenizer = load_moses_detokenizer()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)
    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    if len(tensor_info["inputs"]) != 2:
        raise ValueError(f"Expected 2 model inputs, got {len(tensor_info['inputs'])}")
    if len(tensor_info["outputs"]) != 1:
        raise ValueError(f"Expected 1 model output, got {len(tensor_info['outputs'])}")

    src_input_shape = tuple(int(value) for value in tensor_info["inputs"][0]["dims"])
    prev_input_shape = tuple(int(value) for value in tensor_info["inputs"][1]["dims"])
    output_shape = tuple(int(value) for value in tensor_info["outputs"][0]["dims"])

    source_length = src_input_shape[-1]
    target_length = prev_input_shape[-1]
    target_vocab_size = output_shape[-1]

    if source_length != MAX_TEXT_LENGTH:
        raise ValueError(f"Model source length is {source_length}, expected {MAX_TEXT_LENGTH}")
    if target_length != MAX_TEXT_LENGTH:
        raise ValueError(f"Model target length is {target_length}, expected {MAX_TEXT_LENGTH}")
    if target_vocab_size != len(target_symbols):
        raise ValueError(
            f"Model output vocabulary size is {target_vocab_size}, "
            f"target dictionary size is {len(target_symbols)}"
        )

    target_pad_id = target_indices[PAD_TOKEN]
    target_eos_id = target_indices[EOS_TOKEN]

    print(f"Source input shape: {src_input_shape}")
    print(f"Previous-output input shape: {prev_input_shape}")
    print(f"Output shape: {output_shape}")
    print(f"Source PAD ID: {source_indices[PAD_TOKEN]}")
    print(f"Source EOS ID: {source_indices[EOS_TOKEN]}")
    print(f"Source UNK ID: {source_indices[UNK_TOKEN]}")
    print(f"Target PAD ID: {target_pad_id}")
    print(f"Target EOS ID: {target_eos_id}")
    print()

    for text_idx, text in enumerate(args.texts, 1):
        print("=" * 60)
        print(f"Translating text {text_idx}/{len(args.texts)}: {text}")
        print("=" * 60)

        try:
            src_tokens, tokenized_text, bpe_text, source_token_ids = preprocess_text(
                text,
                source_indices,
                source_length,
                bpe,
                tokenizer,
            )

            generated_ids = translate(
                amlnn,
                src_tokens,
                target_length,
                target_vocab_size,
                target_pad_id,
                target_eos_id,
            )

            target_bpe_text, translated_text = decode_tokens(
                generated_ids,
                target_symbols,
                detokenizer,
            )

            print(f"Tokenized input: {tokenized_text}")
            print(f"BPE input: {bpe_text}")
            print(f"Source token IDs: {source_token_ids}")
            print(f"Generated token IDs: {generated_ids}")
            print(f"Generated BPE: {target_bpe_text}")
            print(f"Translation: {translated_text}")
        except Exception as error:
            print(f"Error translating text {text_idx}: {error}")

        print()

    print("=" * 60)
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()


if __name__ == "__main__":
    main()
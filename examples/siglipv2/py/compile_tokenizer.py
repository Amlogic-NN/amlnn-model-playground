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
import json
import os
import struct


MAGIC = b"SLIPBPE1"
VERSION = 1


def read_merge(merge):
    if isinstance(merge, str):
        parts = merge.split(" ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid merge string: {merge}")
        return parts[0], parts[1]

    if isinstance(merge, list) and len(merge) >= 2:
        if not isinstance(merge[0], str) or not isinstance(merge[1], str):
            raise ValueError(f"Invalid merge array: {merge}")
        return merge[0], merge[1]

    raise ValueError(f"Unsupported merge format: {merge}")


def write_u32(f, value):
    f.write(struct.pack("<I", int(value)))


def write_u64(f, value):
    f.write(struct.pack("<Q", int(value)))


def write_i32(f, value):
    f.write(struct.pack("<i", int(value)))


def write_string(f, value):
    data = value.encode("utf-8")
    write_u32(f, len(data))
    f.write(data)


def main():
    parser = argparse.ArgumentParser(
        description="Compile SigLIP2 tokenizer.json into a binary tokenizer"
    )

    parser.add_argument(
        "--tokenizer-json",
        required=True,
        help="Path to tokenizer.json"
    )

    parser.add_argument(
        "--output",
        default="./data_bin/siglip_tokenizer.bin",
        help="Output binary path"
    )

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer_json}")

    with open(args.tokenizer_json, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    model = tokenizer.get("model")
    if not isinstance(model, dict):
        raise ValueError("tokenizer.json does not contain a valid model")

    if model.get("type") != "BPE":
        raise ValueError(
            f"Expected BPE tokenizer, got {model.get('type')}"
        )

    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.json model.vocab is not an object")

    merges = model.get("merges", [])
    if not isinstance(merges, list):
        raise ValueError("tokenizer.json model.merges is not an array")

    # ------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------

    pad_id = 0
    eos_id = 1
    bos_id = 2
    unk_id = 3

    for token in tokenizer.get("added_tokens", []):
        if not isinstance(token, dict):
            continue

        token_id = token.get("id")
        content = token.get("content")

        if not isinstance(token_id, int):
            continue

        if content == "<pad>":
            pad_id = token_id
        elif content == "<eos>":
            eos_id = token_id
        elif content == "<bos>":
            bos_id = token_id
        elif content == "<unk>":
            unk_id = token_id

    # ------------------------------------------------------------
    # Convert vocab to int IDs
    # ------------------------------------------------------------

    token_to_id = {}

    for token, token_id in vocab.items():
        if not isinstance(token_id, int):
            raise ValueError(
                f"Invalid vocabulary ID for token {token!r}"
            )

        token_to_id[token] = token_id

    # ------------------------------------------------------------
    # Build merge table.
    #
    # Runtime only needs:
    #
    #   left_id + right_id -> merged_id
    #
    # The merge ordering is represented by the order in the
    # tokenizer.json merge list. Lower rank wins.
    # ------------------------------------------------------------

    compiled_merges = []

    for rank, merge in enumerate(merges):
        left, right = read_merge(merge)

        if left not in token_to_id:
            raise ValueError(
                f"Merge component not found in vocab: {left!r}"
            )

        if right not in token_to_id:
            raise ValueError(
                f"Merge component not found in vocab: {right!r}"
            )

        merged = left + right

        if merged not in token_to_id:
            raise ValueError(
                f"Merged token not found in vocab: "
                f"{left!r} + {right!r} -> {merged!r}"
            )

        left_id = token_to_id[left]
        right_id = token_to_id[right]
        merged_id = token_to_id[merged]

        compiled_merges.append(
            (
                left_id,
                right_id,
                merged_id,
                rank
            )
        )

    # ------------------------------------------------------------
    # Compile only the symbols needed to initialize the BPE.
    #
    # Runtime needs token -> ID lookup for UTF-8 symbols and byte
    # fallback tokens.
    #
    # Writing the complete vocabulary is still useful because the
    # tokenizer may encounter arbitrary Unicode characters.
    # ------------------------------------------------------------

    output_dir = os.path.dirname(
        os.path.abspath(args.output)
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Vocabulary entries: {len(token_to_id):,}")
    print(f"BPE merges:        {len(compiled_merges):,}")
    print(f"PAD ID:            {pad_id}")
    print(f"EOS ID:            {eos_id}")
    print(f"BOS ID:            {bos_id}")
    print(f"UNK ID:            {unk_id}")

    with open(args.output, "wb") as f:

        # Header
        f.write(MAGIC)
        write_u32(f, VERSION)

        # Special token IDs
        write_i32(f, pad_id)
        write_i32(f, eos_id)
        write_i32(f, bos_id)
        write_i32(f, unk_id)

        # Fixed tokenizer properties
        #
        # Normalizer:
        #   " " -> "▁"
        #
        # Post processor:
        #   append <eos>
        #
        f.write(struct.pack("<B", 1))  # replace_space_with_underscore = true
        f.write(struct.pack("<B", 1))  # add_eos = true
        f.write(struct.pack("<B", 1))  # byte_fallback = true
        f.write(struct.pack("<B", 1))  # fuse_unk = true

        # Vocabulary
        write_u64(f, len(token_to_id))

        for token, token_id in token_to_id.items():
            write_i32(f, token_id)
            write_string(f, token)

        # Merge table
        write_u64(f, len(compiled_merges))

        for left_id, right_id, merged_id, rank in compiled_merges:
            write_i32(f, left_id)
            write_i32(f, right_id)
            write_i32(f, merged_id)
            write_i32(f, rank)

    file_size = os.path.getsize(args.output)

    print()
    print(f"Compiled tokenizer: {args.output}")
    print(f"Binary size:        {file_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
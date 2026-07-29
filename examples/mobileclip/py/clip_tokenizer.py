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

"""CLIP BPE tokenizer backed by vocab.json and merges.txt."""

from __future__ import annotations

import json
from pathlib import Path


class CLIPTokenizer:
    def __init__(self) -> None:
        self._byte_to_unicode: dict[int, str] = {}
        self._token_to_id: dict[str, int] = {}
        self._bpe_ranks: dict[tuple[str, str], int] = {}
        self._sot_token_id = 49406
        self._eot_token_id = 49407
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, vocab_path: str | Path, merges_path: str | Path) -> None:
        self._init_byte_to_unicode()

        with open(vocab_path, encoding="utf-8") as file:
            vocab = json.load(file)
        for token, token_id in vocab.items():
            self._token_to_id[token] = int(token_id)

        if "<|startoftext|>" in self._token_to_id:
            self._sot_token_id = self._token_to_id["<|startoftext|>"]
        if "<|endoftext|>" in self._token_to_id:
            self._eot_token_id = self._token_to_id["<|endoftext|>"]

        with open(merges_path, encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        rank = 0
        start = 1 if lines and lines[0].startswith("#version") else 0
        for line in lines[start:]:
            parts = line.split()
            if len(parts) >= 2:
                self._bpe_ranks[(parts[0], parts[1])] = rank
                rank += 1

        self._loaded = True

    def load_from_dir(self, tokenizer_dir: str | Path) -> None:
        base = Path(tokenizer_dir).expanduser().resolve()
        vocab_path = base / "vocab.json"
        merges_path = base / "merges.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"vocab.json not found in tokenizer dir: {base}")
        if not merges_path.exists():
            raise FileNotFoundError(f"merges.txt not found in tokenizer dir: {base}")
        self.load(vocab_path, merges_path)

    def encode(self, text: str, max_len: int = 77) -> list[int]:
        if not self._loaded:
            raise RuntimeError("Tokenizer not loaded")

        tokens = [self._sot_token_id]
        for word in self._pre_tokenize(text):
            unicode_word = self._bytes_to_unicode_str(word)
            for piece in self._bpe(unicode_word):
                token_id = self._token_to_id.get(piece)
                if token_id is None and piece.endswith("</w>"):
                    token_id = self._token_to_id.get(piece[:-4])
                if token_id is not None:
                    tokens.append(token_id)

        tokens.append(self._eot_token_id)

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            tokens[-1] = self._eot_token_id

        while len(tokens) < max_len:
            tokens.append(0)

        return tokens

    def _init_byte_to_unicode(self) -> None:
        bs = list(range(33, 127))
        bs.extend(range(161, 173))
        bs.extend(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self._byte_to_unicode = {
            byte: chr(codepoint) for byte, codepoint in zip(bs, cs)
        }

    def _bytes_to_unicode_str(self, raw: str) -> str:
        return "".join(self._byte_to_unicode[byte] for byte in raw.encode("latin-1"))

    def _bpe(self, token: str) -> list[str]:
        word = [char + ("</w>" if index == len(token) - 1 else "")
                for index, char in enumerate(token)]
        if len(word) == 1:
            return word

        while True:
            if len(word) < 2:
                break

            best_rank = None
            best_idx = -1
            for index in range(len(word) - 1):
                rank = self._bpe_ranks.get((word[index], word[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = index

            if best_idx == -1:
                break

            merged = word[best_idx] + word[best_idx + 1]
            word = word[:best_idx] + [merged] + word[best_idx + 2:]

        return word

    def _pre_tokenize(self, text: str) -> list[str]:
        cleaned = "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in text)
        words: list[str] = []
        current = ""

        for char in cleaned:
            if char in {" ", "\t", "\n", "\r"}:
                if current:
                    words.append(current)
                    current = ""
                continue

            is_alpha_or_digit = ("a" <= char <= "z") or ("0" <= char <= "9")
            cur_is_alpha = bool(current) and (
                ("a" <= current[-1] <= "z") or ("0" <= current[-1] <= "9")
            )

            if current and not is_alpha_or_digit and cur_is_alpha:
                words.append(current)
                current = ""
            elif current and is_alpha_or_digit and not cur_is_alpha:
                words.append(current)
                current = ""

            current += char

        if current:
            words.append(current)

        return words

/*
 * Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SIGLIP_TOKENIZER_H
#define SIGLIP_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct SigLIPMerge
{
    int32_t left_id;
    int32_t right_id;
    int32_t merged_id;
    int32_t rank;
};

class SigLIPTokenizer
{
public:
    SigLIPTokenizer() = default;

    // Loads the precompiled tokenizer from:
    // ./data_bin/siglip_tokenizer.bin
    //
    // tokenizer_dir is retained for API compatibility.
    bool load_from_dir(const std::string &tokenizer_dir);

    std::vector<int64_t> encode(
        const std::string &text,
        int max_len = 64) const;

    bool is_loaded() const
    {
        return loaded_;
    }

private:
    bool loaded_ = false;

    int32_t pad_id_ = 0;
    int32_t eos_id_ = 1;
    int32_t bos_id_ = 2;
    int32_t unk_id_ = 3;

    bool replace_space_ = true;
    bool add_eos_ = true;
    bool byte_fallback_ = true;
    bool fuse_unk_ = true;

    // UTF-8 character / byte-fallback symbol -> token ID
    std::unordered_map<std::string, int32_t> token_to_id_;

    // Pair of token IDs -> merge index/rank.
    //
    // key:
    //   (left_id << 32) | right_id
    //
    // value:
    //   index into merges_
    std::unordered_map<uint64_t, uint32_t> merge_lookup_;

    std::vector<SigLIPMerge> merges_;

    static uint64_t make_pair_key(
        int32_t left,
        int32_t right);

    int get_utf8_char_len(
        unsigned char c) const;

    std::vector<std::string> utf8_split(
        const std::string &text) const;

    std::vector<std::string> byte_fallback(
        const std::string &character) const;

    std::vector<int32_t> apply_bpe(
        const std::vector<int32_t> &symbols) const;

    bool load_binary(
        const std::string &path);
};

#endif // SIGLIP_TOKENIZER_H
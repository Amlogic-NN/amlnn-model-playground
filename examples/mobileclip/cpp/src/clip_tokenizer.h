/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
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

#ifndef CLIP_TOKENIZER_H
#define CLIP_TOKENIZER_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <climits>
#include <unordered_map>

class CLIPTokenizer {
public:
    CLIPTokenizer() = default;

    bool load(const std::string& vocab_path, const std::string& merges_path);
    bool load_from_dir(const std::string& tokenizer_dir);

    // Encode text to token IDs (with SOT/EOT and padding to max_len)
    std::vector<int64_t> encode(const std::string& text, int max_len = 77) const;

    bool is_loaded() const { return loaded_; }
    size_t vocab_size() const { return token_to_id_.size(); }

private:
    using BPEPair = std::pair<std::string, std::string>;

    std::unordered_map<uint8_t, char32_t> byte_to_unicode_;
    std::unordered_map<char32_t, uint8_t> unicode_to_byte_;

    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;

    std::map<BPEPair, int> bpe_ranks_;

    int sot_token_id_ = 49406;  // <|startoftext|>
    int eot_token_id_ = 49407;  // <|endoftext|>

    bool loaded_ = false;

    void init_byte_to_unicode();
    static std::vector<char32_t> utf8_to_codepoints(const std::string& str);
    static std::string codepoints_to_utf8(const std::vector<char32_t>& cps);
    std::vector<std::string> bpe(const std::string& token) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    std::string bytes_to_unicode_str(const std::string& raw) const;
};

#endif // CLIP_TOKENIZER_H
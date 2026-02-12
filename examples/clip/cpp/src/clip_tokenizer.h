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

#ifndef CLIP_TOKENIZER_H
#define CLIP_TOKENIZER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

class CLIPTokenizer {
public:
    CLIPTokenizer() = default;

    /**
     * Load tokenizer from vocab.json and merges.txt
     * @param vocab_path   Path to vocab.json
     * @param merges_path  Path to merges.txt
     * @return true on success
     */
    bool load(const std::string& vocab_path, const std::string& merges_path);

    /**
     * Load tokenizer from a directory containing vocab.json and merges.txt
     * @param tokenizer_dir  Path to directory
     * @return true on success
     */
    bool load_from_dir(const std::string& tokenizer_dir);

    /**
     * Tokenize text to token IDs with padding/truncation.
     * Adds <|startoftext|> and <|endoftext|> automatically.
     *
     * @param text      Input text string
     * @param max_len   Maximum sequence length (default: 64)
     * @return Vector of int64_t token IDs with shape [max_len]
     */
    std::vector<int64_t> encode(const std::string& text, int max_len = 64) const;

    /**
     * Check if tokenizer is loaded
     */
    bool is_loaded() const { return loaded_; }

    /**
     * Get vocabulary size
     */
    size_t vocab_size() const { return token_to_id_.size(); }

private:
    // BPE pair
    using BPEPair = std::pair<std::string, std::string>;

    // Byte-to-unicode mapping (GPT-2 style)
    std::unordered_map<uint8_t, char32_t> byte_to_unicode_;
    std::unordered_map<char32_t, uint8_t> unicode_to_byte_;

    // Vocabulary
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;

    // BPE merge rules (pair -> priority rank)
    std::map<BPEPair, int> bpe_ranks_;

    // Special token IDs
    int sot_token_id_ = 49406;  // <|startoftext|>
    int eot_token_id_ = 49407;  // <|endoftext|>

    bool loaded_ = false;

    // Initialize byte-to-unicode mapping
    void init_byte_to_unicode();

    // Convert UTF-8 string to vector of unicode codepoints
    static std::vector<char32_t> utf8_to_codepoints(const std::string& str);

    // Convert unicode codepoints to UTF-8 string
    static std::string codepoints_to_utf8(const std::vector<char32_t>& cps);

    // Apply BPE to a single word (already converted to unicode representation)
    std::vector<std::string> bpe(const std::string& token) const;

    // Clean and split text using CLIP's regex pattern
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Convert raw bytes to unicode string using byte_to_unicode mapping
    std::string bytes_to_unicode_str(const std::string& raw) const;
};

#endif // CLIP_TOKENIZER_H


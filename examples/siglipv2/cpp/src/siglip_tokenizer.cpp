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

#include "siglip_tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
constexpr char MAGIC[8] = {'S','L','I','P','B','P','E','1'};
constexpr uint32_t BINARY_VERSION = 1;

bool read_u32(std::ifstream &f, uint32_t &v) {
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    return f.good();
}

bool read_u64(std::ifstream &f, uint64_t &v) {
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    return f.good();
}

bool read_i32(std::ifstream &f, int32_t &v) {
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    return f.good();
}

bool read_string(std::ifstream &f, std::string &s) {
    uint32_t len = 0;
    if (!read_u32(f, len)) return false;
    s.resize(len);
    if (len > 0) {
        f.read(s.data(), len);
        if (!f.good()) return false;
    }
    return true;
}
}

uint64_t SigLIPTokenizer::make_pair_key(int32_t left, int32_t right) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
           static_cast<uint32_t>(right);
}

int SigLIPTokenizer::get_utf8_char_len(unsigned char c) const {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

std::vector<std::string> SigLIPTokenizer::utf8_split(const std::string &text) const {
    std::vector<std::string> result;
    size_t i = 0;
    while (i < text.size()) {
        int len = get_utf8_char_len(static_cast<unsigned char>(text[i]));
        if (i + static_cast<size_t>(len) > text.size()) len = 1;
        result.emplace_back(text.substr(i, static_cast<size_t>(len)));
        i += static_cast<size_t>(len);
    }
    return result;
}

std::vector<std::string> SigLIPTokenizer::byte_fallback(const std::string &character) const {
    std::vector<std::string> result;
    for (unsigned char byte : std::vector<unsigned char>(character.begin(), character.end())) {
        std::ostringstream ss;
        ss << "<0x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(byte) << ">";
        result.push_back(ss.str());
    }
    return result;
}

bool SigLIPTokenizer::load_binary(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open compiled tokenizer: " << path << std::endl;
        return false;
    }

    char magic[8] = {};
    file.read(magic, sizeof(magic));
    if (!file.good() || std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
        std::cerr << "Invalid SigLIP tokenizer binary." << std::endl;
        return false;
    }

    uint32_t version = 0;
    if (!read_u32(file, version) || version != BINARY_VERSION) {
        std::cerr << "Unsupported tokenizer binary version: " << version << std::endl;
        return false;
    }

    if (!read_i32(file, pad_id_) || !read_i32(file, eos_id_) ||
        !read_i32(file, bos_id_) || !read_i32(file, unk_id_)) {
        std::cerr << "Failed to read tokenizer special tokens." << std::endl;
        return false;
    }

    uint8_t flag = 0;
    file.read(reinterpret_cast<char *>(&flag), sizeof(flag));
    if (!file.good()) return false;
    replace_space_ = flag != 0;

    file.read(reinterpret_cast<char *>(&flag), sizeof(flag));
    if (!file.good()) return false;
    add_eos_ = flag != 0;

    file.read(reinterpret_cast<char *>(&flag), sizeof(flag));
    if (!file.good()) return false;
    byte_fallback_ = flag != 0;

    file.read(reinterpret_cast<char *>(&flag), sizeof(flag));
    if (!file.good()) return false;
    fuse_unk_ = flag != 0;

    uint64_t vocab_size = 0;
    if (!read_u64(file, vocab_size)) return false;

    token_to_id_.clear();
    token_to_id_.reserve(static_cast<size_t>(vocab_size * 1.3));

    for (uint64_t i = 0; i < vocab_size; ++i) {
        int32_t id = 0;
        std::string token;
        if (!read_i32(file, id) || !read_string(file, token)) {
            std::cerr << "Failed reading vocabulary entry " << i << std::endl;
            return false;
        }
        token_to_id_.emplace(std::move(token), id);
    }

    uint64_t merge_count = 0;
    if (!read_u64(file, merge_count)) return false;

    merges_.clear();
    merges_.reserve(static_cast<size_t>(merge_count));
    merge_lookup_.clear();
    merge_lookup_.reserve(static_cast<size_t>(merge_count * 1.3));

    for (uint64_t i = 0; i < merge_count; ++i) {
        SigLIPMerge merge{};
        if (!read_i32(file, merge.left_id) || !read_i32(file, merge.right_id) ||
            !read_i32(file, merge.merged_id) || !read_i32(file, merge.rank)) {
            std::cerr << "Failed reading merge " << i << std::endl;
            return false;
        }

        uint32_t index = static_cast<uint32_t>(merges_.size());
        merges_.push_back(merge);
        merge_lookup_[make_pair_key(merge.left_id, merge.right_id)] = index;
    }

    loaded_ = true;
    return true;
}

bool SigLIPTokenizer::load_from_dir(const std::string &tokenizer_dir) {
    (void)tokenizer_dir;

    const std::string binary_path = "./data_bin/siglip_tokenizer.bin";
    if (!load_binary(binary_path)) {
        std::cerr << "Failed to load compiled tokenizer: " << binary_path << std::endl;
        std::cerr << "Run compile_tokenizer.py first." << std::endl;
        return false;
    }

    std::cout << "[Info] Compiled SigLIP2 tokenizer loaded successfully." << std::endl;
    std::cout << "[Info] Vocabulary size: " << token_to_id_.size() << std::endl;
    std::cout << "[Info] BPE merges: " << merges_.size() << std::endl;
    std::cout << "[Info] Special tokens: pad=" << pad_id_
              << ", eos=" << eos_id_ << ", bos=" << bos_id_
              << ", unk=" << unk_id_ << std::endl;
    return true;
}

std::vector<int32_t> SigLIPTokenizer::apply_bpe(const std::vector<int32_t> &input) const {
    std::vector<int32_t> symbols = input;

    while (symbols.size() > 1) {
        int32_t best_rank = INT_MAX;
        int best_index = -1;
        int32_t best_merged_id = -1;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_lookup_.find(make_pair_key(symbols[i], symbols[i + 1]));
            if (it == merge_lookup_.end()) continue;

            const SigLIPMerge &merge = merges_[it->second];
            if (merge.rank < best_rank) {
                best_rank = merge.rank;
                best_index = static_cast<int>(i);
                best_merged_id = merge.merged_id;
            }
        }

        if (best_index < 0) break;

        std::vector<int32_t> next;
        next.reserve(symbols.size());

        for (size_t i = 0; i < symbols.size();) {
            if (static_cast<int>(i) == best_index) {
                next.push_back(best_merged_id);
                i += 2;
            } else {
                next.push_back(symbols[i]);
                ++i;
            }
        }

        symbols.swap(next);
    }

    return symbols;
}

std::vector<int64_t> SigLIPTokenizer::encode(const std::string &text, int max_len) const {
    if (!loaded_) {
        std::cerr << "Tokenizer not loaded!" << std::endl;
        return std::vector<int64_t>(max_len, pad_id_);
    }

    if (max_len <= 0) return {};

    std::string normalized;
    normalized.reserve(text.size() + 16);

    const std::string space_marker = "\xE2\x96\x81";
    for (char c : text) {
        if (replace_space_ && c == ' ')
            normalized += space_marker;
        else
            normalized.push_back(c);
    }

    const std::vector<std::string> chars = utf8_split(normalized);
    std::vector<int32_t> symbols;
    symbols.reserve(chars.size());

    for (const std::string &character : chars) {
        auto it = token_to_id_.find(character);
        if (it != token_to_id_.end()) {
            symbols.push_back(it->second);
            continue;
        }

        if (byte_fallback_) {
            const auto fallback = byte_fallback(character);
            for (const std::string &byte_token : fallback) {
                auto byte_it = token_to_id_.find(byte_token);
                if (byte_it != token_to_id_.end()) {
                    symbols.push_back(byte_it->second);
                } else if (!fuse_unk_ || symbols.empty() || symbols.back() != unk_id_) {
                    symbols.push_back(unk_id_);
                }
            }
        } else if (!fuse_unk_ || symbols.empty() || symbols.back() != unk_id_) {
            symbols.push_back(unk_id_);
        }
    }

    symbols = apply_bpe(symbols);

    std::vector<int64_t> tokens;
    tokens.reserve(max_len);

    for (int32_t id : symbols)
        tokens.push_back(static_cast<int64_t>(id));

    if (add_eos_) {
        if (static_cast<int>(tokens.size()) < max_len) {
            tokens.push_back(static_cast<int64_t>(eos_id_));
        } else {
            tokens.resize(max_len);
            if (!tokens.empty()) tokens.back() = static_cast<int64_t>(eos_id_);
            return tokens;
        }
    }

    while (static_cast<int>(tokens.size()) < max_len)
        tokens.push_back(static_cast<int64_t>(pad_id_));

    if (static_cast<int>(tokens.size()) > max_len) {
        tokens.resize(max_len);
        if (add_eos_ && !tokens.empty())
            tokens.back() = static_cast<int64_t>(eos_id_);
    }

    return tokens;
}
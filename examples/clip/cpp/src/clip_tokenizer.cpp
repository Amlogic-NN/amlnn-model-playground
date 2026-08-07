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

#include "clip_tokenizer.h"
#include "json.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <set>
#include <cassert>
#include <codecvt>
#include <locale>

using json = nlohmann::ordered_json;

// Reference: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

void CLIPTokenizer::init_byte_to_unicode()
{
    byte_to_unicode_.clear();
    unicode_to_byte_.clear();

    // Printable ASCII ranges that map to themselves
    // '!' (33) to '~' (126), '¡' (161) to '¬' (172), '®' (174) to 'ÿ' (255)
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i)
        bs.push_back(i); // '!' to '~'
    for (int i = 161; i <= 172; ++i)
        bs.push_back(i); // '¡' to '¬'
    for (int i = 174; i <= 255; ++i)
        bs.push_back(i); // '®' to 'ÿ'

    std::vector<int> cs(bs.begin(), bs.end());

    // Map remaining bytes (0-32, 127-160, 173) to 256+
    int n = 0;
    for (int b = 0; b < 256; ++b)
    {
        if (std::find(bs.begin(), bs.end(), b) == bs.end())
        {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i)
    {
        byte_to_unicode_[static_cast<uint8_t>(bs[i])] = static_cast<char32_t>(cs[i]);
        unicode_to_byte_[static_cast<char32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
}

// ========== UTF-8 Helpers ==========

std::vector<char32_t> CLIPTokenizer::utf8_to_codepoints(const std::string &str)
{
    std::vector<char32_t> result;
    size_t i = 0;
    while (i < str.size())
    {
        char32_t cp = 0;
        unsigned char c = str[i];
        int len = 0;
        if (c < 0x80)
        {
            cp = c;
            len = 1;
        }
        else if ((c & 0xE0) == 0xC0)
        {
            cp = c & 0x1F;
            len = 2;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            cp = c & 0x0F;
            len = 3;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            cp = c & 0x07;
            len = 4;
        }
        else
        {
            ++i;
            continue;
        }
        for (int j = 1; j < len && (i + j) < str.size(); ++j)
        {
            cp = (cp << 6) | (str[i + j] & 0x3F);
        }
        result.push_back(cp);
        i += len;
    }
    return result;
}

std::string CLIPTokenizer::codepoints_to_utf8(const std::vector<char32_t> &cps)
{
    std::string result;
    for (char32_t cp : cps)
    {
        if (cp < 0x80)
        {
            result += static_cast<char>(cp);
        }
        else if (cp < 0x800)
        {
            result += static_cast<char>(0xC0 | (cp >> 6));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else if (cp < 0x10000)
        {
            result += static_cast<char>(0xE0 | (cp >> 12));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
        else
        {
            result += static_cast<char>(0xF0 | (cp >> 18));
            result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return result;
}

// ========== Load Functions ==========

bool CLIPTokenizer::load(const std::string &vocab_path, const std::string &merges_path)
{
    init_byte_to_unicode();

    // Load vocab.json
    {
        std::ifstream file(vocab_path);
        if (!file.is_open())
        {
            std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
            return false;
        }

        try
        {
            json j;
            file >> j;
            for (auto it = j.begin(); it != j.end(); ++it)
            {
                std::string token = it.key();
                int id = it.value().get<int>();
                token_to_id_[token] = id;
                id_to_token_[id] = token;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error parsing vocab.json: " << e.what() << std::endl;
            return false;
        }
    }

    // Find special token IDs
    if (token_to_id_.count("<|startoftext|>"))
    {
        sot_token_id_ = token_to_id_["<|startoftext|>"];
    }
    if (token_to_id_.count("<|endoftext|>"))
    {
        eot_token_id_ = token_to_id_["<|endoftext|>"];
    }

    // Load merges.txt
    {
        std::ifstream file(merges_path);
        if (!file.is_open())
        {
            std::cerr << "Failed to open merges file: " << merges_path << std::endl;
            return false;
        }

        std::string line;
        int rank = 0;

        // Skip header line "#version: ..." if present
        if (std::getline(file, line))
        {
            if (line.find("#version") == std::string::npos)
            {
                // First line is not a header, process it
                std::istringstream iss(line);
                std::string a, b;
                if (iss >> a >> b)
                {
                    bpe_ranks_[{a, b}] = rank++;
                }
            }
        }

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            std::istringstream iss(line);
            std::string a, b;
            if (iss >> a >> b)
            {
                bpe_ranks_[{a, b}] = rank++;
            }
        }
    }

    loaded_ = true;
    printf("[Info] CLIPTokenizer loaded: vocab_size=%zu, merges=%zu\n",
           token_to_id_.size(), bpe_ranks_.size());
    return true;
}

bool CLIPTokenizer::load_from_dir(const std::string &tokenizer_dir)
{
    std::string dir = tokenizer_dir;
    // Ensure trailing slash
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
    {
        dir += "/";
    }
    return load(dir + "vocab.json", dir + "merges.txt");
}

// ========== BPE Implementation ==========

std::string CLIPTokenizer::bytes_to_unicode_str(const std::string &raw) const
{
    std::vector<char32_t> result;
    for (unsigned char c : raw)
    {
        auto it = byte_to_unicode_.find(c);
        if (it != byte_to_unicode_.end())
        {
            result.push_back(it->second);
        }
    }
    return codepoints_to_utf8(result);
}

std::vector<std::string> CLIPTokenizer::bpe(const std::string &token) const
{
    // Convert token to individual unicode characters as strings
    auto codepoints = utf8_to_codepoints(token);
    if (codepoints.empty())
        return {};

    // Each character becomes a separate piece
    std::vector<std::string> word;
    for (size_t i = 0; i < codepoints.size(); ++i)
    {
        std::string piece = codepoints_to_utf8({codepoints[i]});
        // CLIP adds </w> to the last character
        if (i == codepoints.size() - 1)
        {
            piece += "</w>";
        }
        word.push_back(piece);
    }

    if (word.size() == 1)
        return word;

    // Iteratively merge the most frequent pairs
    while (true)
    {
        if (word.size() < 2)
            break;

        // Find the pair with the lowest rank
        int best_rank = INT_MAX;
        int best_idx = -1;

        for (size_t i = 0; i < word.size() - 1; ++i)
        {
            auto it = bpe_ranks_.find({word[i], word[i + 1]});
            if (it != bpe_ranks_.end() && it->second < best_rank)
            {
                best_rank = it->second;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx == -1)
            break; // No more merges possible

        // Merge the pair at best_idx
        std::string merged = word[best_idx] + word[best_idx + 1];
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ++i)
        {
            if (static_cast<int>(i) == best_idx)
            {
                new_word.push_back(merged);
                ++i; // Skip next element
            }
            else
            {
                new_word.push_back(word[i]);
            }
        }
        word = new_word;
    }

    return word;
}

std::vector<std::string> CLIPTokenizer::pre_tokenize(const std::string &text) const
{
    // CLIP tokenizer: lowercase + basic clean + split by pattern
    // Pattern from CLIP: <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
    // Simplified version for ASCII-dominant text:

    std::string cleaned;
    // Lowercase and basic whitespace normalization
    for (char c : text)
    {
        if (c >= 'A' && c <= 'Z')
        {
            cleaned += (c - 'A' + 'a');
        }
        else
        {
            cleaned += c;
        }
    }

    // Simple tokenization: split by whitespace and punctuation
    std::vector<std::string> words;
    std::string current;

    for (size_t i = 0; i < cleaned.size(); ++i)
    {
        char c = cleaned[i];

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        {
            if (!current.empty())
            {
                words.push_back(current);
                current.clear();
            }
            // Add space prefix to next word (CLIP uses space-prefixed tokens)
            if (i + 1 < cleaned.size() && cleaned[i + 1] != ' ')
            {
                // Next word will get a space prefix via the byte encoding
            }
        }
        else
        {
            // Check if punctuation should be separate token
            bool is_alpha_or_digit = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
            bool cur_is_alpha = !current.empty() &&
                                ((current.back() >= 'a' && current.back() <= 'z') ||
                                 (current.back() >= '0' && current.back() <= '9'));

            if (!current.empty() && !is_alpha_or_digit && cur_is_alpha)
            {
                // Start new token for punctuation
                words.push_back(current);
                current.clear();
            }
            else if (!current.empty() && is_alpha_or_digit && !cur_is_alpha)
            {
                words.push_back(current);
                current.clear();
            }
            current += c;
        }
    }
    if (!current.empty())
    {
        words.push_back(current);
    }

    return words;
}

// ========== Encode ==========

std::vector<int64_t> CLIPTokenizer::encode(const std::string &text, int max_len) const
{
    if (!loaded_)
    {
        std::cerr << "Tokenizer not loaded!" << std::endl;
        return std::vector<int64_t>(max_len, 0);
    }

    std::vector<int64_t> tokens;

    // Add start-of-text token
    tokens.push_back(sot_token_id_);

    // Pre-tokenize
    std::vector<std::string> words = pre_tokenize(text);

    // Process each word
    for (const auto &word : words)
    {
        // Convert raw bytes to unicode representation
        std::string unicode_word = bytes_to_unicode_str(word);

        // Apply BPE
        std::vector<std::string> bpe_tokens = bpe(unicode_word);

        // Look up token IDs
        for (const auto &bt : bpe_tokens)
        {
            auto it = token_to_id_.find(bt);
            if (it != token_to_id_.end())
            {
                tokens.push_back(it->second);
            }
            else
            {
                // Unknown token, try without </w>
                std::string no_ew = bt;
                if (no_ew.size() >= 4 && no_ew.substr(no_ew.size() - 4) == "</w>")
                {
                    no_ew = no_ew.substr(0, no_ew.size() - 4);
                }
                auto it2 = token_to_id_.find(no_ew);
                if (it2 != token_to_id_.end())
                {
                    tokens.push_back(it2->second);
                }
                // else: skip unknown token
            }
        }
    }

    // Add end-of-text token
    tokens.push_back(eot_token_id_);

    // Truncate if necessary
    if (static_cast<int>(tokens.size()) > max_len)
    {
        tokens.resize(max_len);
        // Ensure EOT is at the end
        tokens.back() = eot_token_id_;
    }

    // Pad to max_len with EOT token (consistent with HuggingFace CLIPTokenizer)
    while (static_cast<int>(tokens.size()) < max_len)
    {
        tokens.push_back(eot_token_id_);
    }

    return tokens;
}

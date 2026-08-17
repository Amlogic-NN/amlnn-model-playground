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

#include "post_process_whisper.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "pre_post_common.h"

namespace
{
constexpr int MULTILINGUAL_VOCAB_SIZE = 51865;
constexpr int TOKEN_EOT = 50257;
constexpr int TOKEN_SOT = 50258;
constexpr int TOKEN_TRANSLATE = 50358;
constexpr int TOKEN_TRANSCRIBE = 50359;
constexpr int TOKEN_SOLM = 50360;
constexpr int TOKEN_PREV = 50361;
constexpr int TOKEN_NOSP = 50362;
constexpr int TOKEN_NOTIMESTAMPS = 50363;
constexpr int TOKEN_TIMESTAMP_BEGIN = 50364;

std::string trim_text(const std::string &text)
{
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return "";
    }

    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::vector<std::string> split_words(const std::string &text)
{
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;

    while (stream >> word)
    {
        words.push_back(word);
    }

    return words;
}

std::string normalize_word(const std::string &word)
{
    std::string normalized;

    for (unsigned char character : word)
    {
        if (std::isalnum(character) || character == '\'')
        {
            normalized.push_back(static_cast<char>(std::tolower(character)));
        }
    }

    return normalized;
}

std::string join_words(const std::vector<std::string> &words)
{
    std::string text;

    for (const std::string &word : words)
    {
        if (!text.empty())
        {
            text.push_back(' ');
        }

        text += word;
    }

    return text;
}
}

whisper_vocab read_token_info(const std::string &token_path)
{
    whisper_vocab vocab;
    std::ifstream fin(token_path, std::ios::binary);

    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, token_path.c_str());
        return {};
    }

    int32_t stored_vocab_size = 0;
    fin.read(reinterpret_cast<char *>(&stored_vocab_size), sizeof(stored_vocab_size));

    if (!fin || stored_vocab_size <= 0)
    {
        fprintf(stderr, "%s: invalid tokenizer header in '%s'\n", __func__, token_path.c_str());
        return {};
    }

    if (stored_vocab_size < TOKEN_EOT)
    {
        fprintf(
            stderr,
            "%s: tokenizer contains %d base entries; multilingual Whisper requires at least %d\n",
            __func__,
            stored_vocab_size,
            TOKEN_EOT
        );
        return {};
    }

    for (int32_t token_id = 0; token_id < stored_vocab_size; ++token_id)
    {
        uint32_t length = 0;
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));

        if (!fin)
        {
            fprintf(stderr, "%s: failed while reading token %d\n", __func__, token_id);
            return {};
        }

        std::string token(length, '\0');

        if (length > 0)
        {
            fin.read(token.data(), length);

            if (!fin)
            {
                fprintf(stderr, "%s: failed while reading token %d data\n", __func__, token_id);
                return {};
            }
        }

        vocab.token_to_id[token] = token_id;
        vocab.id_to_token[token_id] = token;
    }

    // The deployed model is the multilingual Whisper vocabulary.
    vocab.n_vocab = MULTILINGUAL_VOCAB_SIZE;
    vocab.token_eot = TOKEN_EOT;
    vocab.token_sot = TOKEN_SOT;
    vocab.token_translate = TOKEN_TRANSLATE;
    vocab.token_transcribe = TOKEN_TRANSCRIBE;
    vocab.token_solm = TOKEN_SOLM;
    vocab.token_prev = TOKEN_PREV;
    vocab.token_nosp = TOKEN_NOSP;
    vocab.token_not = TOKEN_NOTIMESTAMPS;
    vocab.token_beg = TOKEN_TIMESTAMP_BEGIN;

    return vocab;
}

std::string decode_tokens(const std::vector<int64_t> &token_ids, const whisper_vocab &vocab)
{
    std::string transcription;

    for (int64_t token_id : token_ids)
    {
        if (token_id < 0 || token_id >= vocab.token_eot)
        {
            continue;
        }

        const auto token = vocab.id_to_token.find(static_cast<int32_t>(token_id));
        if (token != vocab.id_to_token.end())
        {
            transcription += token->second;
        }
    }

    return trim_text(transcription);
}

std::string merge_transcriptions(const std::vector<std::string> &transcriptions)
{
    std::string combined;

    for (const std::string &raw_transcription : transcriptions)
    {
        const std::string transcription = trim_text(raw_transcription);

        if (transcription.empty())
        {
            continue;
        }

        if (combined.empty())
        {
            combined = transcription;
            continue;
        }

        std::vector<std::string> previous_words = split_words(combined);
        std::vector<std::string> current_words = split_words(transcription);
        const size_t max_overlap = std::min<size_t>(
            20,
            std::min(previous_words.size(), current_words.size())
        );

        size_t matched_words = 0;

        for (size_t count = max_overlap; count > 0; --count)
        {
            bool matches = true;

            for (size_t index = 0; index < count; ++index)
            {
                const std::string previous_word = normalize_word(
                    previous_words[previous_words.size() - count + index]
                );
                const std::string current_word = normalize_word(current_words[index]);

                if (previous_word.empty() || previous_word != current_word)
                {
                    matches = false;
                    break;
                }
            }

            if (matches)
            {
                matched_words = count;
                break;
            }
        }

        if (matched_words > 0)
        {
            previous_words.insert(
                previous_words.end(),
                current_words.begin() + matched_words,
                current_words.end()
            );
            combined = join_words(previous_words);
            continue;
        }

        const std::string previous_last = normalize_word(previous_words.back());
        const std::string current_first = normalize_word(current_words.front());

        if (previous_last.size() >= 3 &&
            current_first.compare(0, previous_last.size(), previous_last) == 0)
        {
            previous_words.pop_back();
            previous_words.insert(previous_words.end(), current_words.begin(), current_words.end());
            combined = join_words(previous_words);
        }
        else if (current_first.size() >= 3 &&
                 previous_last.compare(0, current_first.size(), current_first) == 0)
        {
            previous_words.insert(previous_words.end(), current_words.begin() + 1, current_words.end());
            combined = join_words(previous_words);
        }
        else
        {
            combined += " " + transcription;
        }
    }

    return trim_text(combined);
}
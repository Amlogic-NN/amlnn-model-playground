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

#ifndef TEXT_PROCESSING_H
#define TEXT_PROCESSING_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

const int MAX_TEXT_LENGTH = 64;
const int EXPECTED_VOCAB_SIZE = 44512;

const std::string BOS_TOKEN = "<s>";
const std::string PAD_TOKEN = "<pad>";
const std::string EOS_TOKEN = "</s>";
const std::string UNK_TOKEN = "<unk>";

struct Dictionary
{
    std::vector<std::string> symbols;
    std::unordered_map<std::string, int64_t> indices;
};

struct TextInput
{
    std::vector<int64_t> tensor;
    std::vector<int64_t> token_ids;
    std::string tokenized_text;
    std::string bpe_text;
};

struct DecodedText
{
    std::string bpe_text;
    std::string text;
};

class BPEProcessor
{
public:
    bool load(const std::string &codes_path);
    std::string process_line(const std::string &line) const;

private:
    struct PairHash
    {
        size_t operator()(const std::pair<std::string, std::string> &value) const;
    };

    int version = 1;
    std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> codes;

    std::vector<std::string> process_word(const std::string &word) const;
};

Dictionary load_dictionary(const std::string &dict_path);
std::string moses_tokenize(const std::string &text);
TextInput preprocess_text(const std::string &text, const Dictionary &source_dictionary,
                          const BPEProcessor &bpe, int max_length);
DecodedText decode_tokens(const std::vector<int64_t> &token_ids,
                          const Dictionary &target_dictionary);

#endif
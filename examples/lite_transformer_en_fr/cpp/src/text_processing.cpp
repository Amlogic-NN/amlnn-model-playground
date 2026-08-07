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

#include "text_processing.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

static void replace_all(std::string &text, const std::string &from, const std::string &to)
{
    if (from.empty())
        return;

    size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos)
    {
        text.replace(position, from.size(), to);
        position += to.size();
    }
}

static std::string trim(const std::string &text)
{
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])))
        ++start;

    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1])))
        --end;

    return text.substr(start, end - start);
}

static std::vector<std::string> split_whitespace(const std::string &text)
{
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;

    while (stream >> token)
        tokens.push_back(token);

    return tokens;
}

static std::string join_tokens(const std::vector<std::string> &tokens)
{
    std::string text;

    for (size_t i = 0; i < tokens.size(); ++i)
    {
        if (i > 0)
            text += " ";
        text += tokens[i];
    }

    return text;
}

static std::vector<std::string> split_utf8(const std::string &text)
{
    std::vector<std::string> characters;

    for (size_t i = 0; i < text.size();)
    {
        unsigned char value = static_cast<unsigned char>(text[i]);
        size_t length = 1;

        if ((value & 0xE0) == 0xC0)
            length = 2;
        else if ((value & 0xF0) == 0xE0)
            length = 3;
        else if ((value & 0xF8) == 0xF0)
            length = 4;

        if (i + length > text.size())
            length = 1;

        characters.push_back(text.substr(i, length));
        i += length;
    }

    return characters;
}

static std::string xml_escape(const std::string &token)
{
    std::string escaped = token;
    replace_all(escaped, "&", "&amp;");
    replace_all(escaped, "|", "&#124;");
    replace_all(escaped, "<", "&lt;");
    replace_all(escaped, ">", "&gt;");
    replace_all(escaped, "'", "&apos;");
    replace_all(escaped, "\"", "&quot;");
    return escaped;
}

static std::string xml_unescape(const std::string &token)
{
    std::string unescaped = token;
    replace_all(unescaped, "&quot;", "\"");
    replace_all(unescaped, "&apos;", "'");
    replace_all(unescaped, "&lt;", "<");
    replace_all(unescaped, "&gt;", ">");
    replace_all(unescaped, "&#124;", "|");
    replace_all(unescaped, "&amp;", "&");
    return unescaped;
}

static bool is_ascii_letter_or_digit(char value)
{
    return std::isalnum(static_cast<unsigned char>(value)) != 0;
}

static bool is_separate_punctuation(char value)
{
    const std::string punctuation = ".,!?;:%()[]{}<>\"/\\";
    return punctuation.find(value) != std::string::npos;
}

static bool ends_with(const std::string &text, const std::string &suffix)
{
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool starts_with(const std::string &text, const std::string &prefix)
{
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

static std::vector<std::string> remove_bpe_markers(const std::vector<std::string> &tokens)
{
    std::vector<std::string> words;
    std::string current_word;

    for (const std::string &token : tokens)
    {
        if (ends_with(token, "@@"))
        {
            current_word += token.substr(0, token.size() - 2);
        }
        else
        {
            current_word += token;
            words.push_back(current_word);
            current_word.clear();
        }
    }

    if (!current_word.empty())
        words.push_back(current_word);

    return words;
}

static std::string moses_detokenize_french(const std::vector<std::string> &tokens)
{
    std::string result;
    bool join_next = false;

    for (std::string token : tokens)
    {
        token = xml_unescape(token);

        if (token == "@-@")
            token = "-";

        if (result.empty())
        {
            result = token;
            join_next = token == "-";
            continue;
        }

        bool no_space_before =
            token == "." || token == "," || token == ")" || token == "]" || token == "}" ||
            token == "%" || token == ":" || token == ";" || token == "!" || token == "?" ||
            starts_with(token, "'");

        bool previous_opens =
            ends_with(result, "(") || ends_with(result, "[") || ends_with(result, "{") ||
            ends_with(result, "\"") || ends_with(result, "'");

        if (token == "-")
        {
            result += "-";
            join_next = true;
        }
        else if (join_next || no_space_before || previous_opens)
        {
            result += token;
            join_next = false;
        }
        else
        {
            result += " " + token;
        }
    }

    return result;
}

size_t BPEProcessor::PairHash::operator()(const std::pair<std::string, std::string> &value) const
{
    size_t first_hash = std::hash<std::string>{}(value.first);
    size_t second_hash = std::hash<std::string>{}(value.second);
    return first_hash ^ (second_hash + 0x9e3779b9 + (first_hash << 6) + (first_hash >> 2));
}

bool BPEProcessor::load(const std::string &codes_path)
{
    std::ifstream file(codes_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open BPE codes: " << codes_path << std::endl;
        return false;
    }

    codes.clear();
    version = 1;

    std::string line;
    size_t rank = 0;
    bool first_nonempty_line = true;

    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        line = trim(line);
        if (line.empty())
            continue;

        if (first_nonempty_line && line == "#version: 0.2")
        {
            version = 2;
            first_nonempty_line = false;
            continue;
        }

        first_nonempty_line = false;

        if (!line.empty() && line[0] == '#')
            continue;

        std::istringstream stream(line);
        std::string first;
        std::string second;

        if (!(stream >> first >> second))
            continue;

        std::pair<std::string, std::string> pair = {first, second};
        if (codes.find(pair) == codes.end())
            codes[pair] = rank;

        ++rank;
    }

    if (codes.empty())
    {
        std::cerr << "No BPE merge rules found in: " << codes_path << std::endl;
        return false;
    }

    return true;
}

std::vector<std::string> BPEProcessor::process_word(const std::string &word) const
{
    if (word.empty())
        return {};

    std::vector<std::string> symbols = split_utf8(word);
    std::vector<std::string> encoded;

    if (version == 2)
    {
        encoded = symbols;
        encoded.back() += "</w>";
    }
    else
    {
        encoded = symbols;
        encoded.push_back("</w>");
    }

    while (encoded.size() > 1)
    {
        size_t best_rank = std::numeric_limits<size_t>::max();
        std::pair<std::string, std::string> best_pair;
        bool found = false;

        for (size_t i = 0; i + 1 < encoded.size(); ++i)
        {
            std::pair<std::string, std::string> pair = {encoded[i], encoded[i + 1]};
            auto iterator = codes.find(pair);

            if (iterator != codes.end() && iterator->second < best_rank)
            {
                best_rank = iterator->second;
                best_pair = pair;
                found = true;
            }
        }

        if (!found)
            break;

        std::vector<std::string> merged;
        for (size_t i = 0; i < encoded.size();)
        {
            if (i + 1 < encoded.size() &&
                encoded[i] == best_pair.first &&
                encoded[i + 1] == best_pair.second)
            {
                merged.push_back(encoded[i] + encoded[i + 1]);
                i += 2;
            }
            else
            {
                merged.push_back(encoded[i]);
                ++i;
            }
        }

        encoded.swap(merged);
    }

    if (!encoded.empty() && encoded.back() == "</w>")
        encoded.pop_back();
    else if (!encoded.empty() && ends_with(encoded.back(), "</w>"))
        encoded.back().erase(encoded.back().size() - 4);

    for (size_t i = 0; i + 1 < encoded.size(); ++i)
        encoded[i] += "@@";

    return encoded;
}

std::string BPEProcessor::process_line(const std::string &line) const
{
    std::vector<std::string> input_tokens = split_whitespace(line);
    std::vector<std::string> output_tokens;

    for (const std::string &token : input_tokens)
    {
        std::vector<std::string> pieces = process_word(token);
        output_tokens.insert(output_tokens.end(), pieces.begin(), pieces.end());
    }

    return join_tokens(output_tokens);
}

Dictionary load_dictionary(const std::string &dict_path)
{
    std::ifstream file(dict_path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open dictionary: " + dict_path);

    Dictionary dictionary;
    dictionary.symbols = {BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN};

    for (size_t i = 0; i < dictionary.symbols.size(); ++i)
        dictionary.indices[dictionary.symbols[i]] = static_cast<int64_t>(i);

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        line = trim(line);
        if (line.empty())
            continue;

        size_t separator = line.find_last_of(' ');
        std::string token = separator == std::string::npos ? line : line.substr(0, separator);

        if (dictionary.indices.find(token) == dictionary.indices.end())
        {
            int64_t index = static_cast<int64_t>(dictionary.symbols.size());
            dictionary.indices[token] = index;
            dictionary.symbols.push_back(token);
        }
    }

    return dictionary;
}

std::string moses_tokenize(const std::string &text)
{
    std::string normalized = text;

    replace_all(normalized, "\xE2\x80\x98", "'");
    replace_all(normalized, "\xE2\x80\x99", "'");
    replace_all(normalized, "\xE2\x80\x9C", "\"");
    replace_all(normalized, "\xE2\x80\x9D", "\"");
    replace_all(normalized, "\xE2\x80\x93", "-");
    replace_all(normalized, "\xE2\x80\x94", "-");

    std::string separated;

    for (size_t i = 0; i < normalized.size(); ++i)
    {
        char value = normalized[i];
        char previous = i > 0 ? normalized[i - 1] : '\0';
        char next = i + 1 < normalized.size() ? normalized[i + 1] : '\0';

        if (value == '-' && is_ascii_letter_or_digit(previous) && is_ascii_letter_or_digit(next))
        {
            separated += " @-@ ";
        }
        else if (value == '\'' && is_ascii_letter_or_digit(previous) && is_ascii_letter_or_digit(next))
        {
            separated += " '";
        }
        else if (is_separate_punctuation(value))
        {
            bool decimal_point = (value == '.' || value == ',') &&
                                 std::isdigit(static_cast<unsigned char>(previous)) &&
                                 std::isdigit(static_cast<unsigned char>(next));

            if (decimal_point)
                separated += value;
            else
                separated += std::string(" ") + value + " ";
        }
        else
        {
            separated += value;
        }
    }

    std::vector<std::string> tokens = split_whitespace(separated);
    for (std::string &token : tokens)
        token = xml_escape(token);

    return join_tokens(tokens);
}

TextInput preprocess_text(const std::string &text, const Dictionary &source_dictionary,
                          const BPEProcessor &bpe, int max_length)
{
    int64_t pad_id = source_dictionary.indices.at(PAD_TOKEN);
    int64_t eos_id = source_dictionary.indices.at(EOS_TOKEN);
    int64_t unk_id = source_dictionary.indices.at(UNK_TOKEN);

    TextInput result;
    result.tokenized_text = moses_tokenize(text);
    result.bpe_text = bpe.process_line(result.tokenized_text);

    std::vector<std::string> tokens = split_whitespace(result.bpe_text);
    for (const std::string &token : tokens)
    {
        auto iterator = source_dictionary.indices.find(token);
        result.token_ids.push_back(
            iterator == source_dictionary.indices.end() ? unk_id : iterator->second
        );
    }

    if (static_cast<int>(result.token_ids.size()) >= max_length)
        result.token_ids.resize(max_length - 1);

    result.token_ids.push_back(eos_id);

    result.tensor.assign(max_length, pad_id);
    std::copy(result.token_ids.begin(), result.token_ids.end(), result.tensor.begin());

    return result;
}

DecodedText decode_tokens(const std::vector<int64_t> &token_ids,
                          const Dictionary &target_dictionary)
{
    std::vector<std::string> bpe_tokens;

    for (int64_t token_id : token_ids)
    {
        std::string token = UNK_TOKEN;

        if (token_id >= 0 && token_id < static_cast<int64_t>(target_dictionary.symbols.size()))
            token = target_dictionary.symbols[token_id];

        if (token == BOS_TOKEN || token == PAD_TOKEN || token == EOS_TOKEN)
            continue;

        bpe_tokens.push_back(token);
    }

    DecodedText result;
    result.bpe_text = join_tokens(bpe_tokens);
    std::vector<std::string> words = remove_bpe_markers(bpe_tokens);
    result.text = moses_detokenize_french(words);
    return result;
}
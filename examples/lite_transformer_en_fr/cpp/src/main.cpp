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

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include "postprocess.h"
#include "text_processing.h"
#include "nnsdk2.h"
#include "model_loader.h"

const int DEFAULT_MAX_NEW_TOKENS = 64;
namespace fs = std::filesystem;

static void print_shape(const std::string &name, const std::vector<int> &shape)
{
    std::cout << name << ": [";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i + 1 < shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

static void print_token_ids(const std::string &name, const std::vector<int64_t> &token_ids)
{
    std::cout << name << ": [";
    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        std::cout << token_ids[i];
        if (i + 1 < token_ids.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0]
                  << " <model.adla> <en-fr_text_assets> [--max-new-tokens N] <texts ...>\n";
        std::cout << "Example: " << argv[0]
                  << " lite_transformer.adla en-fr_text_assets \"I am not French.\"\n";
        return 0;
    }

    std::string model_path = argv[1];
    fs::path assets_dir = argv[2];
    int max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
    std::vector<std::string> texts;

    for (int i = 3; i < argc; ++i)
    {
        std::string argument = argv[i];

        if (argument == "--max-new-tokens")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--max-new-tokens requires a value" << std::endl;
                return -1;
            }

            max_new_tokens = std::stoi(argv[++i]);
        }
        else
        {
            texts.push_back(argument);
        }
    }

    if (texts.empty())
    {
        std::cerr << "No input texts supplied" << std::endl;
        return -1;
    }

    fs::path source_dict_path = assets_dir / "dict.en.txt";
    fs::path target_dict_path = assets_dir / "dict.fr.txt";
    fs::path bpe_codes_path = assets_dir / "bpecodes";

    if (!fs::is_regular_file(source_dict_path) ||
        !fs::is_regular_file(target_dict_path) ||
        !fs::is_regular_file(bpe_codes_path))
    {
        std::cerr << "Expected dict.en.txt, dict.fr.txt, and bpecodes in: "
                  << assets_dir << std::endl;
        return -1;
    }

    std::cout << "Lite Transformer English-to-French Demo" << std::endl;

    Dictionary source_dictionary;
    Dictionary target_dictionary;
    BPEProcessor bpe;

    try
    {
        source_dictionary = load_dictionary(source_dict_path.string());
        target_dictionary = load_dictionary(target_dict_path.string());
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        return -1;
    }

    if (!bpe.load(bpe_codes_path.string()))
        return -1;

    std::cout << "Source vocabulary size: " << source_dictionary.symbols.size() << std::endl;
    std::cout << "Target vocabulary size: " << target_dictionary.symbols.size() << std::endl;

    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    if (io_num.n_input != 2 || io_num.n_output != 1)
    {
        std::cerr << "Expected 2 inputs and 1 output, got "
                  << io_num.n_input << " inputs and " << io_num.n_output << " outputs" << std::endl;
        uninit_network(context);
        return -1;
    }

    amlnn_tensor_attr src_attr = query_input_attr(context, 0);
    amlnn_tensor_attr prev_attr = query_input_attr(context, 1);
    amlnn_tensor_attr output_attr = query_output_attr(context, 0);

    if (src_attr.type != AMLNN_TENSOR_INT64 || prev_attr.type != AMLNN_TENSOR_INT64)
    {
        std::cerr << "Both model inputs must be INT64" << std::endl;
        uninit_network(context);
        return -1;
    }

    if (output_attr.type != AMLNN_TENSOR_FLOAT32)
    {
        std::cerr << "Model output must be FLOAT32" << std::endl;
        uninit_network(context);
        return -1;
    }

    std::vector<int> src_shape = get_tensor_shape(src_attr);
    std::vector<int> prev_shape = get_tensor_shape(prev_attr);
    std::vector<int> output_shape = get_tensor_shape(output_attr);

    print_shape("Source input shape", src_shape);
    print_shape("Previous-output input shape", prev_shape);
    print_shape("Output shape", output_shape);

    if (src_shape.empty() || prev_shape.empty() || output_shape.size() != 2)
    {
        std::cerr << "Unexpected Lite Transformer tensor shapes" << std::endl;
        uninit_network(context);
        return -1;
    }

    int source_length = src_shape.back();
    int target_length = prev_shape.back();
    int output_length = output_shape[0];
    int target_vocab_size = output_shape[1];

    if (source_length != MAX_TEXT_LENGTH ||
        target_length != MAX_TEXT_LENGTH ||
        output_length != target_length)
    {
        std::cerr << "Expected source/output token length " << MAX_TEXT_LENGTH << std::endl;
        uninit_network(context);
        return -1;
    }

    if (src_attr.n_elems != static_cast<size_t>(source_length) ||
        prev_attr.n_elems != static_cast<size_t>(target_length))
    {
        std::cerr << "Unexpected input element count" << std::endl;
        uninit_network(context);
        return -1;
    }

    if (source_dictionary.symbols.size() != EXPECTED_VOCAB_SIZE ||
        target_dictionary.symbols.size() != EXPECTED_VOCAB_SIZE ||
        target_vocab_size != EXPECTED_VOCAB_SIZE)
    {
        std::cerr << "Expected vocabulary size " << EXPECTED_VOCAB_SIZE
                  << ", got source=" << source_dictionary.symbols.size()
                  << ", target=" << target_dictionary.symbols.size()
                  << ", model=" << target_vocab_size << std::endl;
        uninit_network(context);
        return -1;
    }

    int64_t target_pad_id = target_dictionary.indices.at(PAD_TOKEN);
    int64_t target_eos_id = target_dictionary.indices.at(EOS_TOKEN);

    std::cout << "Source PAD ID: " << source_dictionary.indices.at(PAD_TOKEN) << std::endl;
    std::cout << "Source EOS ID: " << source_dictionary.indices.at(EOS_TOKEN) << std::endl;
    std::cout << "Source UNK ID: " << source_dictionary.indices.at(UNK_TOKEN) << std::endl;
    std::cout << "Target PAD ID: " << target_pad_id << std::endl;
    std::cout << "Target EOS ID: " << target_eos_id << std::endl
              << std::endl;

    std::vector<amlnn_output> outData(io_num.n_output);

    for (size_t text_index = 0; text_index < texts.size(); ++text_index)
    {
        std::cout << "============================================================" << std::endl;
        std::cout << "Translation " << text_index + 1 << "/" << texts.size() << std::endl;
        std::cout << "============================================================" << std::endl;

        TextInput source_input = preprocess_text(
            texts[text_index], source_dictionary, bpe, source_length);

        std::vector<int64_t> prev_output_tokens(target_length, target_pad_id);
        prev_output_tokens[0] = target_eos_id;

        std::vector<int64_t> generated_ids;
        int max_steps = std::min(max_new_tokens, target_length);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_steps; ++step)
        {
            // The two buffers correspond to serving_default_src_tokens:0
            // and serving_default_prev_output_tokens:0.
            std::vector<void *> input_ptrs = {
                static_cast<void *>(source_input.tensor.data()),
                static_cast<void *>(prev_output_tokens.data())};

            std::vector<size_t> input_sizes = {
                source_input.tensor.size() * sizeof(int64_t),
                prev_output_tokens.size() * sizeof(int64_t)};

            if (!run_multi_input_network(context, input_ptrs, input_sizes, outData))
            {
                std::cerr << "Failed to run network" << std::endl;
                uninit_network(context);
                return -1;
            }

            if (outData.empty() || outData[0].buf == nullptr)
            {
                std::cerr << "Model returned no output" << std::endl;
                uninit_network(context);
                return -1;
            }

            float *logits = reinterpret_cast<float *>(outData[0].buf);
            int64_t next_token_id = greedy_next_token(logits, step, target_vocab_size);

            if (next_token_id == target_eos_id)
                break;

            generated_ids.push_back(next_token_id);

            if (step + 1 < target_length)
                prev_output_tokens[step + 1] = next_token_id;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;

        DecodedText decoded = decode_tokens(generated_ids, target_dictionary);

        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
        std::cout << "Input: " << texts[text_index] << std::endl;
        std::cout << "Tokenized input: " << source_input.tokenized_text << std::endl;
        std::cout << "BPE input: " << source_input.bpe_text << std::endl;
        print_token_ids("Source token IDs", source_input.token_ids);
        print_token_ids("Generated token IDs", generated_ids);
        std::cout << "Generated BPE: " << decoded.bpe_text << std::endl;
        std::cout << "Translation: " << decoded.text << std::endl
                  << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    uninit_network(context);
    return 0;
}
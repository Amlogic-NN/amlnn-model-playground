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

#include "whisper_invoke.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>

#include "post_process_whisper.h"

namespace
{
#define BILLION 1000000000ULL

constexpr int TENSOR_TYPE_FLOAT = 0;
constexpr int TENSOR_TYPE_INT8 = 2;
constexpr int TENSOR_TYPE_UINT8 = 3;
constexpr int TENSOR_TYPE_INT16 = 4;

constexpr int64_t TOKEN_EOT = 50257;
constexpr int64_t TOKEN_SOT = 50258;
constexpr int64_t TOKEN_ENGLISH = 50259;
constexpr int64_t TOKEN_TRANSCRIBE = 50359;
constexpr int64_t TOKEN_NOTIMESTAMPS = 50363;

struct PreparedTensorInput
{
    std::vector<float> float_data;
    std::vector<int8_t> int8_data;
    std::vector<uint8_t> uint8_data;
    std::vector<int16_t> int16_data;

    void *data = nullptr;
    size_t size = 0;
};

uint64_t get_time_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_nsec) + static_cast<uint64_t>(ts.tv_sec) * BILLION;
}

using QueryType = decltype(AMLNN_QUERY_INPUT_ATTR);

bool query_tensor_attr(
    void *context,
    QueryType query_type,
    int index,
    amlnn_tensor_attr &attr,
    const char *label)
{
    memset(&attr, 0, sizeof(attr));
    attr.index = index;

    const int ret = amlnn_query(context, query_type, &attr, sizeof(attr));
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_query failed for %s index %d.\n", label, index);
        return false;
    }

    return true;
}

bool prepare_float_input(
    const std::vector<float> &source,
    const amlnn_tensor_attr &attr,
    PreparedTensorInput &prepared)
{
    if (source.size() != static_cast<size_t>(attr.n_elems))
    {
        printf(
            "Input '%s' contains %zu elements, expected %u.\n",
            attr.name,
            source.size(),
            attr.n_elems
        );
        return false;
    }

    if (attr.type == TENSOR_TYPE_FLOAT)
    {
        prepared.float_data = source;
        prepared.data = prepared.float_data.data();
        prepared.size = prepared.float_data.size() * sizeof(float);
        return true;
    }

    if (attr.type != TENSOR_TYPE_INT8 &&
        attr.type != TENSOR_TYPE_UINT8 &&
        attr.type != TENSOR_TYPE_INT16)
    {
        printf("Unsupported input tensor type %d for '%s'.\n", attr.type, attr.name);
        return false;
    }

    if (attr.scale == 0.0f)
    {
        printf("Input tensor '%s' has an invalid quantization scale of 0.\n", attr.name);
        return false;
    }

    if (attr.type == TENSOR_TYPE_INT8)
    {
        prepared.int8_data.resize(source.size());

        for (size_t i = 0; i < source.size(); ++i)
        {
            const int64_t value = static_cast<int64_t>(std::llround(source[i] / attr.scale + attr.zp));
            prepared.int8_data[i] = static_cast<int8_t>(
                std::clamp<int64_t>(value, -128, 127)
            );
        }

        prepared.data = prepared.int8_data.data();
        prepared.size = prepared.int8_data.size() * sizeof(int8_t);
        return true;
    }

    if (attr.type == TENSOR_TYPE_UINT8)
    {
        prepared.uint8_data.resize(source.size());

        for (size_t i = 0; i < source.size(); ++i)
        {
            const int64_t value = static_cast<int64_t>(std::llround(source[i] / attr.scale + attr.zp));
            prepared.uint8_data[i] = static_cast<uint8_t>(
                std::clamp<int64_t>(value, 0, 255)
            );
        }

        prepared.data = prepared.uint8_data.data();
        prepared.size = prepared.uint8_data.size() * sizeof(uint8_t);
        return true;
    }

    prepared.int16_data.resize(source.size());

    for (size_t i = 0; i < source.size(); ++i)
    {
        const int64_t value = static_cast<int64_t>(std::llround(source[i] / attr.scale + attr.zp));
        prepared.int16_data[i] = static_cast<int16_t>(
            std::clamp<int64_t>(value, -32768, 32767)
        );
    }

    prepared.data = prepared.int16_data.data();
    prepared.size = prepared.int16_data.size() * sizeof(int16_t);
    return true;
}

bool get_float_output(
    void *context,
    const amlnn_tensor_attr &output_attr,
    std::vector<float> &output,
    const char *label)
{
    amlnn_output out_data;
    memset(&out_data, 0, sizeof(out_data));
    out_data.is_float = 1;
    out_data.index = output_attr.index;

    const int ret = amlnn_outputs_get(context, 1, &out_data);
    if (ret != AMLNN_SUCCESS || out_data.buf == nullptr)
    {
        printf("amlnn_outputs_get failed for %s.\n", label);
        return false;
    }

    const size_t expected_size = static_cast<size_t>(output_attr.n_elems) * sizeof(float);
    if (out_data.size != expected_size)
    {
        printf(
            "%s output contains %zu bytes, expected %zu bytes.\n",
            label,
            out_data.size,
            expected_size
        );
        return false;
    }

    const float *output_ptr = reinterpret_cast<const float *>(out_data.buf);
    output.assign(output_ptr, output_ptr + output_attr.n_elems);
    return true;
}
}

void *init_network_file(const char *model_path)
{
    void *context = nullptr;

    amlnn_init_config init_config;
    memset(&init_config, 0, sizeof(amlnn_init_config));
    init_config.backend_type = AMLNN_BACKEND_ADLA_NPU;

    const int ret = amlnn_init(&context, const_cast<char *>(model_path), 0, &init_config);
    if (ret != AMLNN_SUCCESS || context == nullptr)
    {
        printf("amlnn_init failed for %s.\n", model_path);
        return nullptr;
    }

    return context;
}

bool query_whisper_model_info(
    void *encoder_context,
    void *decoder_context,
    WhisperModelInfo &model_info)
{
    if (!query_tensor_attr(
            encoder_context,
            AMLNN_QUERY_INPUT_ATTR,
            0,
            model_info.encoder_input,
            "encoder input"))
    {
        return false;
    }

    if (!query_tensor_attr(
            encoder_context,
            AMLNN_QUERY_OUTPUT_ATTR,
            0,
            model_info.encoder_output,
            "encoder output"))
    {
        return false;
    }

    if (!query_tensor_attr(
            decoder_context,
            AMLNN_QUERY_INPUT_ATTR,
            0,
            model_info.decoder_ids_input,
            "decoder input"))
    {
        return false;
    }

    if (!query_tensor_attr(
            decoder_context,
            AMLNN_QUERY_INPUT_ATTR,
            1,
            model_info.decoder_hidden_input,
            "decoder input"))
    {
        return false;
    }

    if (!query_tensor_attr(
            decoder_context,
            AMLNN_QUERY_OUTPUT_ATTR,
            0,
            model_info.decoder_output,
            "decoder output"))
    {
        return false;
    }

    if (model_info.encoder_output.n_elems != model_info.decoder_hidden_input.n_elems)
    {
        printf(
            "Encoder output contains %u elements, but decoder hidden input expects %u.\n",
            model_info.encoder_output.n_elems,
            model_info.decoder_hidden_input.n_elems
        );
        return false;
    }

    model_info.decoder_length = static_cast<int>(model_info.decoder_ids_input.n_elems);

    if (model_info.decoder_length <= 0 ||
        model_info.decoder_output.n_elems % model_info.decoder_length != 0)
    {
        printf("Decoder output shape is incompatible with decoder input length.\n");
        return false;
    }

    model_info.vocab_size = static_cast<int>(
        model_info.decoder_output.n_elems / model_info.decoder_length
    );

    if (model_info.vocab_size <= TOKEN_NOTIMESTAMPS)
    {
        printf("Decoder vocabulary size %d is too small for multilingual Whisper.\n", model_info.vocab_size);
        return false;
    }

    return true;
}

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr)
{
    std::vector<int> shape;

    for (uint32_t i = 0; i < attr.n_dims; ++i)
    {
        const int dimension = static_cast<int>(attr.dims[i]);
        if (dimension > 1)
        {
            shape.push_back(dimension);
        }
    }

    if (shape.empty())
    {
        shape.push_back(1);
    }

    return shape;
}

void print_shape(const std::vector<int> &shape)
{
    std::cout << "[";

    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }

        std::cout << shape[i];
    }

    std::cout << "]" << std::endl;
}

void print_tensor_info(const char *label, const amlnn_tensor_attr &attr)
{
    std::cout << label << ": name=" << attr.name << ", shape=";
    print_shape(get_tensor_shape(attr));
    std::cout << "  type=" << attr.type
              << ", elements=" << attr.n_elems
              << ", scale=" << attr.scale
              << ", zp=" << attr.zp
              << std::endl;
}

bool run_network_encoder_process(
    void *context,
    const std::vector<float> &input_features,
    const amlnn_tensor_attr &input_attr,
    const amlnn_tensor_attr &output_attr,
    std::vector<float> &encoder_output)
{
    PreparedTensorInput prepared_input;
    if (!prepare_float_input(input_features, input_attr, prepared_input))
    {
        return false;
    }

    amlnn_input input;
    memset(&input, 0, sizeof(input));
    input.index = input_attr.index;
    input.buf = prepared_input.data;
    input.size = prepared_input.size;

    int ret = amlnn_inputs_set(context, 1, &input);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_inputs_set failed for encoder.\n");
        return false;
    }

    ret = amlnn_run(context, nullptr);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_run failed for encoder.\n");
        return false;
    }

    return get_float_output(context, output_attr, encoder_output, "encoder");
}

bool run_network_decoder(
    void *context,
    const std::vector<float> &encoder_output,
    const WhisperModelInfo &model_info,
    const whisper_vocab &vocab,
    std::string &transcription,
    std::vector<uint64_t> *decoder_inference_times_ms)
{
    PreparedTensorInput prepared_hidden;
    if (!prepare_float_input(
            encoder_output,
            model_info.decoder_hidden_input,
            prepared_hidden))
    {
        return false;
    }

    std::vector<int64_t> active_tokens = {
        TOKEN_SOT,
        TOKEN_ENGLISH,
        TOKEN_TRANSCRIBE,
        TOKEN_NOTIMESTAMPS
    };

    if (active_tokens.size() >= static_cast<size_t>(model_info.decoder_length))
    {
        printf(
            "Decoder prefix contains %zu tokens, but decoder length is %d.\n",
            active_tokens.size(),
            model_info.decoder_length
        );
        return false;
    }

    std::vector<int64_t> generated_tokens;

    while (active_tokens.size() < static_cast<size_t>(model_info.decoder_length))
    {
        std::vector<int64_t> decoder_ids(
            static_cast<size_t>(model_info.decoder_length),
            TOKEN_EOT
        );
        std::copy(active_tokens.begin(), active_tokens.end(), decoder_ids.begin());

        amlnn_input inputs[2];
        memset(inputs, 0, sizeof(inputs));

        inputs[0].index = model_info.decoder_ids_input.index;
        inputs[0].buf = decoder_ids.data();
        inputs[0].size = decoder_ids.size() * sizeof(int64_t);

        inputs[1].index = model_info.decoder_hidden_input.index;
        inputs[1].buf = prepared_hidden.data;
        inputs[1].size = prepared_hidden.size;

        const uint64_t invoke_start_time = get_time_count();

        int ret = amlnn_inputs_set(context, 2, inputs);
        if (ret != AMLNN_SUCCESS)
        {
            printf("amlnn_inputs_set failed for decoder.\n");
            return false;
        }

        ret = amlnn_run(context, nullptr);
        if (ret != AMLNN_SUCCESS)
        {
            printf("amlnn_run failed for decoder.\n");
            return false;
        }

        std::vector<float> logits;
        if (!get_float_output(
                context,
                model_info.decoder_output,
                logits,
                "decoder"))
        {
            return false;
        }

        if (decoder_inference_times_ms != nullptr)
        {
            decoder_inference_times_ms->push_back(
                (get_time_count() - invoke_start_time) / 1000000ULL
            );
        }

        const size_t row_index = active_tokens.size() - 1;
        const size_t begin_index = row_index * static_cast<size_t>(model_info.vocab_size);
        const size_t end_index = begin_index + static_cast<size_t>(model_info.vocab_size);

        if (end_index > logits.size())
        {
            printf("Decoder logits row %zu exceeds output size %zu.\n", row_index, logits.size());
            return false;
        }

        const auto max_value = std::max_element(
            logits.begin() + begin_index,
            logits.begin() + end_index
        );
        const int64_t next_token = static_cast<int64_t>(
            std::distance(logits.begin() + begin_index, max_value)
        );

        generated_tokens.push_back(next_token);

        if (next_token == TOKEN_EOT)
        {
            break;
        }

        active_tokens.push_back(next_token);
    }

    transcription = decode_tokens(generated_tokens, vocab);
    return true;
}

int destroy_network(void *context)
{
    if (context == nullptr)
    {
        return AMLNN_SUCCESS;
    }

    const int ret = amlnn_destroy(context);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_destroy failed.\n");
        return -1;
    }

    return ret;
}
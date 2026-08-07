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

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <vector>

#include "nnsdk2.h"
#include "whisper.h"
#include "whisper_invoke.h"

bool is_finish = false;
static int decoder_input_1_size = 2;         /* init decoder input_1 size*/
whisper_vocab vocab_out_init;

///////////////////////////////////////////////////////////

#define TIKTOKEN_ID_STOP 50256
#define INPUT_SHAPE 48

extern whisper_vocab read_token_info(std::string token_path);

void* init_network_file(const char *model_path)
{
    void *context = nullptr;
    static bool is_worker_initialized = false;

    if (!is_worker_initialized) {
        vocab_out_init = read_token_info("./data_bin/tokenizer_info.bin");
        is_worker_initialized = true;
    }

    amlnn_init_config init_config;
    memset(&init_config, 0, sizeof(amlnn_init_config));
    init_config.backend_type = AMLNN_BACKEND_ADLA_NPU;

    int ret = amlnn_init(&context, (void*)model_path, 0, &init_config);
    if (ret != AMLNN_SUCCESS || context == nullptr)
    {
        printf("amlnn_init fail for %s\n", model_path);
        return nullptr;
    }

    return context;
}

bool is_finish_end() {
    return is_finish;
}

std::vector<float> run_network_encoder_process(void *qcontext, const std::vector<float>& input_ids)
{
    int ret = 0;
    is_finish = false; /* init is_finish -> false */

    // 1. Set Input
    amlnn_input inData;
    memset(&inData, 0, sizeof(amlnn_input));
    inData.index = 0;
    inData.buf = (void*)input_ids.data();
    inData.size = input_ids.size() * sizeof(float);

    ret = amlnn_inputs_set(qcontext, 1, &inData);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_inputs_set fail for encoder.\n");
    }

    // 2. Run Inference
    ret = amlnn_run(qcontext, nullptr);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_run fail for encoder.\n");
    }

    // 3. Get Output
    amlnn_output outData;
    memset(&outData, 0, sizeof(amlnn_output));
    outData.is_float = 1;
    outData.index = 0;

    ret = amlnn_outputs_get(qcontext, 1, &outData);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_outputs_get fail for encoder.\n");
    }

    size_t outputData_size = outData.size / sizeof(float);
    float* out_ptr = reinterpret_cast<float*>(outData.buf);

    // Copy output safely before returning
    std::vector<float> buf_data(out_ptr, out_ptr + outputData_size);

    return buf_data;
}

std::string run_network_decoder(void *qcontext_sec, Input_Decoder* input_data)
{
    int ret = 0;
    int max_index = 0;
    std::string out;

    amlnn_input inData[2];
    memset(inData, 0, sizeof(inData));

    // Query index 0 to see if it wants Tokens (48 elements) or Audio Features
    amlnn_tensor_attr attr0;
    memset(&attr0, 0, sizeof(attr0));
    attr0.index = 0;
    amlnn_query(qcontext_sec, AMLNN_QUERY_INPUT_ATTR, &attr0, sizeof(attr0));

    int token_idx = (attr0.n_elems == input_data->input_1_size) ? 0 : 1;
    int audio_idx = (token_idx == 0) ? 1 : 0;

    // 1. Assign Audio Features (float)
    inData[audio_idx].index = audio_idx;
    inData[audio_idx].buf = (void*)input_data->input_0;
    inData[audio_idx].size = input_data->input_0_size * sizeof(float);

    // 2. Assign Tokens (int64)
    inData[token_idx].index = token_idx;
    inData[token_idx].buf = (void*)input_data->input_1;
    inData[token_idx].size = input_data->input_1_size * sizeof(int64_t);

    ret = amlnn_inputs_set(qcontext_sec, 2, inData);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_inputs_set fail for decoder.\n");
    }

    // Run Inference
    ret = amlnn_run(qcontext_sec, nullptr);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_run fail for decoder.\n");
    }

    // Get Output
    amlnn_output outData;
    memset(&outData, 0, sizeof(amlnn_output));
    outData.is_float = 1;
    outData.index = 0;

    ret = amlnn_outputs_get(qcontext_sec, 1, &outData);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_outputs_get fail for decoder.\n");
    }

    float* buf_data = reinterpret_cast<float*>(outData.buf);

    size_t id_shape = decoder_input_1_size;
    size_t begin_count = (id_shape - 1) * 51864;    // shape [1, 64, 51864]
    size_t last_count = id_shape * 51864 - 1;

    // get max_value and max_index
    auto max_it = std::max_element(buf_data + begin_count, buf_data + last_count);
    max_index = std::distance(buf_data + begin_count, max_it);

    input_data->input_1[id_shape] = max_index;

    if (max_index == TIKTOKEN_ID_STOP || id_shape >= INPUT_SHAPE) {
        is_finish = true;
        if (max_index != TIKTOKEN_ID_STOP)
            out = vocab_out_init.id_to_token.at(max_index).c_str();
        decoder_input_1_size = 2; // Reset
    }
    else {
        out = vocab_out_init.id_to_token.at(max_index).c_str();
        decoder_input_1_size++;
    }

    return out;
}

int destroy_network(void *qcontext)
{
    int ret = amlnn_destroy(qcontext);
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_destroy fail.\n");
        return -1;
    }

    return ret;
}
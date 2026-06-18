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

#include "model_invoke.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "nnsdk2.h"

void* init_network_file(const char *model_path)
{
    void *context = nullptr;
    amlnn_init_config config;
    memset(&config, 0, sizeof(amlnn_init_config));

    config.backend_type = AMLNN_BACKEND_ADLA_NPU;

    int ret = amlnn_init(&context, (void*)model_path, 0, &config);
    if (ret != AMLNN_SUCCESS)
    {
        printf("[Error] amlnn_init failed for %s. Code: %d\n", model_path, ret);
        return nullptr;
    }

    return context;
}

std::vector<float> run_image_model(void* qcontext, const std::vector<float>& input_data)
{
    if (!qcontext || input_data.empty()) return {};

    // 1. Query input attribute for scale and zero_point
    amlnn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    amlnn_query(qcontext, AMLNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));

    // 2. Quantize Input if needed
    std::vector<int8_t> quant_buf_int8;
    std::vector<uint8_t> quant_buf_uint8;
    void* buffer_to_submit = (void*)input_data.data();
    size_t buffer_size = input_data.size() * sizeof(float);

    if (in_attr.type == AMLNN_TENSOR_INT8) {
        quant_buf_int8.resize(in_attr.n_elems);
        for (uint32_t i = 0; i < in_attr.n_elems && i < input_data.size(); ++i) {
            float val = std::round(input_data[i] / in_attr.scale) + in_attr.zp;
            quant_buf_int8[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }
        buffer_to_submit = quant_buf_int8.data();
        buffer_size = quant_buf_int8.size() * sizeof(int8_t);
    }
    else if (in_attr.type == AMLNN_TENSOR_UINT8) {
        quant_buf_uint8.resize(in_attr.n_elems);
        for (uint32_t i = 0; i < in_attr.n_elems && i < input_data.size(); ++i) {
            float val = std::round(input_data[i] / in_attr.scale) + in_attr.zp;
            quant_buf_uint8[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
        buffer_to_submit = quant_buf_uint8.data();
        buffer_size = quant_buf_uint8.size() * sizeof(uint8_t);
    }

    // 3. Set Input
    amlnn_input inData;
    memset(&inData, 0, sizeof(amlnn_input));
    inData.index = 0;
    inData.buf = buffer_to_submit;
    inData.size = buffer_size;

    if (amlnn_inputs_set(qcontext, 1, &inData) != AMLNN_SUCCESS) return {};

    // 4. Run Inference
    if (amlnn_run(qcontext, nullptr) != AMLNN_SUCCESS) return {};

    // 5. Get Output (Ask SDK to automatically dequantize back to float32)
    amlnn_output outData;
    memset(&outData, 0, sizeof(amlnn_output));
    outData.index = 0;
    outData.is_float = 1;

    if (amlnn_outputs_get(qcontext, 1, &outData) != AMLNN_SUCCESS) return {};

    // 6. Copy to vector
    float* output_ptr = reinterpret_cast<float*>(outData.buf);
    size_t output_elements = outData.size / sizeof(float);
    return std::vector<float>(output_ptr, output_ptr + output_elements);
}

std::vector<float> run_text_model(void* qcontext, const std::vector<int64_t>& input_ids)
{
    if (!qcontext || input_ids.empty()) return {};

    // Query text model input type
    amlnn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    amlnn_query(qcontext, AMLNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));

    std::vector<int32_t> downcast_buf;
    void* buffer_to_submit = (void*)input_ids.data();
    size_t buffer_size = input_ids.size() * sizeof(int64_t);

    if (in_attr.type == AMLNN_TENSOR_INT32) {
        downcast_buf.reserve(input_ids.size());
        for (int64_t id : input_ids) {
            downcast_buf.push_back(static_cast<int32_t>(id));
        }
        buffer_to_submit = downcast_buf.data();
        buffer_size = downcast_buf.size() * sizeof(int32_t);
    }

    amlnn_input inData;
    memset(&inData, 0, sizeof(amlnn_input));
    inData.index = 0;
    inData.buf = buffer_to_submit;
    inData.size = buffer_size;

    if (amlnn_inputs_set(qcontext, 1, &inData) != AMLNN_SUCCESS) return {};
    if (amlnn_run(qcontext, nullptr) != AMLNN_SUCCESS) return {};

    amlnn_output outData;
    memset(&outData, 0, sizeof(amlnn_output));
    outData.index = 0;
    outData.is_float = 1; // Extract float32 embeddings

    if (amlnn_outputs_get(qcontext, 1, &outData) != AMLNN_SUCCESS) return {};

    float* output_ptr = reinterpret_cast<float*>(outData.buf);
    size_t output_elements = outData.size / sizeof(float);
    return std::vector<float>(output_ptr, output_ptr + output_elements);
}

int destroy_network(void *qcontext)
{
    if (qcontext == nullptr) return -1;

    int ret = amlnn_destroy(qcontext);
    if (ret != AMLNN_SUCCESS)
    {
        printf("[Error] amlnn_destroy failed. Code: %d\n", ret);
        return -1;
    }
    return 0;
}
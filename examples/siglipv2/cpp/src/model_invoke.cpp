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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include "nnsdk2.h"
#include "model_loader.h"

static std::vector<uint8_t> prepare_input_tensor(const std::vector<float> &input_data, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;
    if (input_data.empty())
        return tensor_data;

    size_t total_elements = input_data.size();

    // FP16 models use float32 host input with this runtime.
    if (attr.type == AMLNN_TENSOR_FLOAT32 || attr.type == AMLNN_TENSOR_FLOAT16)
    {
        tensor_data.resize(total_elements * sizeof(float));
        std::memcpy(tensor_data.data(), input_data.data(), tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(total_elements * sizeof(int16_t));
        for (size_t i = 0; i < total_elements; ++i)
        {
            float quantized = std::round(input_data[i] / attr.scale + attr.zp);
            int16_t value = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, quantized)));
            std::memcpy(tensor_data.data() + i * sizeof(int16_t), &value, sizeof(int16_t));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            float quantized = std::round(input_data[i] / attr.scale + attr.zp);
            tensor_data[i] = static_cast<uint8_t>(static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, quantized))));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements);
        for (size_t i = 0; i < total_elements; ++i)
        {
            float quantized = std::round(input_data[i] / attr.scale + attr.zp);
            tensor_data[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, quantized)));
        }
    }
    else
    {
        std::cerr << "Unsupported image tensor type: " << attr.type << std::endl;
    }

    return tensor_data;
}

static std::vector<uint8_t> prepare_text_tensor(const std::vector<int64_t> &input_ids, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;
    if (input_ids.empty())
        return tensor_data;

    size_t total_elements = input_ids.size();

    if (attr.type == AMLNN_TENSOR_INT64)
    {
        tensor_data.resize(total_elements * sizeof(int64_t));
        std::memcpy(tensor_data.data(), input_ids.data(), tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT32)
    {
        tensor_data.resize(total_elements * sizeof(int32_t));
        for (size_t i = 0; i < total_elements; ++i)
        {
            int32_t value = static_cast<int32_t>(input_ids[i]);
            std::memcpy(tensor_data.data() + i * sizeof(int32_t), &value, sizeof(int32_t));
        }
    }
    else
    {
        std::cerr << "Unsupported text tensor type: " << attr.type << std::endl;
    }

    return tensor_data;
}

std::vector<float> run_image_model(void *context, const std::vector<float> &input_data)
{
    if (!context || input_data.empty())
        return {};

    amlnn_tensor_attr input_attr = query_input_attr(context, 0);
    amlnn_tensor_attr output_attr = query_output_attr(context, 0);

    if (input_data.size() != input_attr.n_elems)
    {
        std::cerr << "image input element mismatch: expected " << input_attr.n_elems << ", got " << input_data.size() << std::endl;
        return {};
    }

    std::vector<uint8_t> prepared_data = prepare_input_tensor(input_data, input_attr);
    if (prepared_data.empty())
        return {};

    std::vector<amlnn_output> outputs(1);
    if (!run_network(context, prepared_data.data(), prepared_data.size(), outputs))
    {
        std::cerr << "Failed to run image network" << std::endl;
        return {};
    }

    if (outputs[0].buf == nullptr)
        return {};

    float *output_ptr = reinterpret_cast<float *>(outputs[0].buf);
    return std::vector<float>(output_ptr, output_ptr + output_attr.n_elems);
}

std::vector<float> run_text_model(void *context, const std::vector<int64_t> &input_ids)
{
    if (!context || input_ids.empty())
        return {};

    amlnn_tensor_attr input_attr = query_input_attr(context, 0);
    amlnn_tensor_attr output_attr = query_output_attr(context, 0);

    if (input_ids.size() != input_attr.n_elems)
    {
        std::cerr << "Text input element mismatch: expected " << input_attr.n_elems << ", got " << input_ids.size() << std::endl;
        return {};
    }

    std::vector<uint8_t> prepared_ids = prepare_text_tensor(input_ids, input_attr);
    if (prepared_ids.empty())
        return {};

    std::vector<amlnn_output> outputs(1);
    if (!run_network(context, prepared_ids.data(), prepared_ids.size(), outputs))
    {
        std::cerr << "Failed to run text network" << std::endl;
        return {};
    }

    if (outputs[0].buf == nullptr)
        return {};

    float *output_ptr = reinterpret_cast<float *>(outputs[0].buf);
    return std::vector<float>(output_ptr, output_ptr + output_attr.n_elems);
}
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
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

// Helper: Robust Image Input Preparation
static std::vector<uint8_t> prepare_input_tensor(const std::vector<float> &float_img, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;
    if (float_img.empty())
    {
        std::cerr << "prepare_input_tensor: Invalid input image data" << std::endl;
        return tensor_data;
    }

    int total_elements = float_img.size();
    const float *src_ptr = float_img.data();

    // FP16 uses float32 input in this hardware, so we treat it identically to FLOAT32
    if (attr.type == AMLNN_TENSOR_FLOAT32 || attr.type == AMLNN_TENSOR_FLOAT16)
    {
        tensor_data.resize(total_elements * sizeof(float));
        std::memcpy(tensor_data.data(), src_ptr, tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(total_elements * sizeof(int16_t));
        int16_t *dst_ptr = reinterpret_cast<int16_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
    }
    else
    {
        std::cerr << "prepare_input_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

// Helper: Text Input Preparation (Handles INT64 to INT32 downcasting)
static std::vector<uint8_t> prepare_text_tensor(const std::vector<int64_t> &input_data, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;
    if (input_data.empty())
        return tensor_data;

    int total_elements = input_data.size();

    if (attr.type == AMLNN_TENSOR_INT32)
    {
        tensor_data.resize(total_elements * sizeof(int32_t));
        int32_t *dst_ptr = reinterpret_cast<int32_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            dst_ptr[i] = static_cast<int32_t>(input_data[i]);
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT64)
    {
        tensor_data.resize(total_elements * sizeof(int64_t));
        std::memcpy(tensor_data.data(), input_data.data(), tensor_data.size());
    }
    else
    {
        // Fallback default to INT32 for text tokens
        tensor_data.resize(total_elements * sizeof(int32_t));
        int32_t *dst_ptr = reinterpret_cast<int32_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            dst_ptr[i] = static_cast<int32_t>(input_data[i]);
        }
    }
    return tensor_data;
}

std::vector<float> run_image_model(void *qcontext, const std::vector<float> &input_data)
{
    if (!qcontext || input_data.empty())
        return {};

    // 1. Query attributes using model_loader
    amlnn_tensor_attr in_attr = query_input_attr(qcontext, 0);
    amlnn_tensor_attr out_attr = query_output_attr(qcontext, 0);

    // 2. Prepare dynamic tensor formatting
    std::vector<uint8_t> prepared_data = prepare_input_tensor(input_data, in_attr);
    if (prepared_data.empty())
        return {};

    // 3. Run Inference using model_loader
    std::vector<amlnn_output> outData(1); // 1 Output expected

    if (!run_network(qcontext, prepared_data.data(), prepared_data.size(), outData))
    {
        std::cerr << "Failed to run image network" << std::endl;
        return {};
    }

    if (outData.empty() || outData[0].buf == nullptr)
        return {};

    // 4. Retrieve auto-dequantized Float32 output
    float *output_ptr = reinterpret_cast<float *>(outData[0].buf);
    size_t output_elements = out_attr.n_elems; // Pull exact size from output_attr

    return std::vector<float>(output_ptr, output_ptr + output_elements);
}

std::vector<float> run_text_model(void *qcontext, const std::vector<int64_t> &input_ids)
{
    if (!qcontext || input_ids.empty())
        return {};

    // 1. Query attributes using model_loader
    amlnn_tensor_attr in_attr_ids = query_input_attr(qcontext, 0);
    amlnn_tensor_attr out_attr = query_output_attr(qcontext, 0);

    // 2. Prepare tensors
    std::vector<uint8_t> prepared_ids = prepare_text_tensor(input_ids, in_attr_ids);

    if (prepared_ids.empty())
        return {};

    std::vector<amlnn_output> outData(1); // 1 Output expected

    if (!run_network(qcontext,
                     prepared_ids.data(), prepared_ids.size(),
                     outData))
    {
        std::cerr << "Failed to run text network" << std::endl;
        return {};
    }

    if (outData.empty() || outData[0].buf == nullptr)
        return {};

    // 4. Retrieve auto-dequantized Float32 output
    float *output_ptr = reinterpret_cast<float *>(outData[0].buf);
    size_t output_elements = out_attr.n_elems;

    return std::vector<float>(output_ptr, output_ptr + output_elements);
}
// -------------------------------------------------------------------------
// Exposed Functions
// -------------------------------------------------------------------------

/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model_loader.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>

#define LOGE(...)                     \
    do                                \
    {                                 \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

int init_network(std::string model_path, void *&qcontext)
{
    amlnn_init_config init_config;
    memset(&init_config, 0, sizeof(amlnn_init_config));
    init_config.backend_type = AMLNN_BACKEND_ADLA_NPU;

    return amlnn_init(&qcontext, (void *)model_path.c_str(), 0, &init_config);
}

int uninit_network(void *qcontext)
{
    int ret = amlnn_destroy(qcontext);
    if (ret)
    {
        LOGE("aml_module_destroy fail.");
        return -1;
    }

    return 0;
}

amlnn_tensor_attr query_input_attr(void *context, uint32_t index)
{
    amlnn_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = index;
    amlnn_query(context, AMLNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    return attr;
}

amlnn_tensor_attr query_output_attr(void *context, uint32_t index)
{
    amlnn_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = index;
    amlnn_query(context, AMLNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
    return attr;
}

// Helper to safely run the network and extract float outputs
bool run_network(void *context, void *input_data, size_t input_size, std::vector<amlnn_output> &outputs)
{
    // Set Input
    amlnn_input inData;
    memset(&inData, 0, sizeof(amlnn_input));
    inData.index = 0;
    inData.buf = input_data;
    inData.size = input_size;

    if (amlnn_inputs_set(context, 1, &inData) != AMLNN_SUCCESS)
        return false;

    // Run Inference
    if (amlnn_run(context, nullptr) != AMLNN_SUCCESS)
        return false;

    // Get Outputs
    for (size_t i = 0; i < outputs.size(); i++)
    {
        memset(&outputs[i], 0, sizeof(amlnn_output));
        outputs[i].is_float = 1;
        outputs[i].index = i;
    }

    if (amlnn_outputs_get(context, outputs.size(), outputs.data()) != AMLNN_SUCCESS)
        return false;

    return true; // Success!
}
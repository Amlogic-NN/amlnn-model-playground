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

#include "postprocess.h"
#include <cstring>

bool run_multi_input_network(void *context,
                             const std::vector<void *> &input_data,
                             const std::vector<size_t> &input_sizes,
                             std::vector<amlnn_output> &outputs)
{
    if (input_data.size() != input_sizes.size())
        return false;

    std::vector<amlnn_input> inData(input_data.size());

    for (size_t i = 0; i < input_data.size(); ++i)
    {
        memset(&inData[i], 0, sizeof(amlnn_input));
        inData[i].index = i;
        inData[i].buf = input_data[i];
        inData[i].size = input_sizes[i];
    }

    if (amlnn_inputs_set(context, inData.size(), inData.data()) != AMLNN_SUCCESS)
        return false;

    if (amlnn_run(context, nullptr) != AMLNN_SUCCESS)
        return false;

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        memset(&outputs[i], 0, sizeof(amlnn_output));
        outputs[i].is_float = 1;
        outputs[i].index = i;
    }

    if (amlnn_outputs_get(context, outputs.size(), outputs.data()) != AMLNN_SUCCESS)
        return false;

    return true;
}

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr)
{
    std::vector<int> shape;

    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
            shape.push_back(attr.dims[i]);
    }

    return shape;
}

int64_t greedy_next_token(const float *logits, int step, int vocabulary_size)
{
    const float *step_logits = logits + static_cast<size_t>(step) * vocabulary_size;
    int64_t best_token = 0;
    float best_score = step_logits[0];

    for (int token = 1; token < vocabulary_size; ++token)
    {
        if (step_logits[token] > best_score)
        {
            best_score = step_logits[token];
            best_token = token;
        }
    }

    return best_token;
}
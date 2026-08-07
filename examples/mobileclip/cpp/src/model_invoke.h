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

#ifndef MODEL_INVOKE_H
#define MODEL_INVOKE_H

#include <vector>
#include <cstdint>
#include <string>

#include "nnsdk2.h"

struct ModelInputTensorDesc {
    uint32_t dim_count = 0;
    uint32_t dims[AMLNN_MAX_DIMS] = {0};
    amlnn_tensor_format tensor_format = AMLNN_TENSOR_NCHW;
    amlnn_tensor_type tensor_type = AMLNN_TENSOR_FLOAT32;
    float scale = 1.0f;
    int32_t zero_point = 0;
};

struct AdlaInitOptions {
    bool enable_neon = true;
    bool enable_openmp = false;
    int openmp_num = 2;
};

void* init_network_file(const char* model_path, const AdlaInitOptions& options = AdlaInitOptions());
bool get_input_tensor_desc(void* qcontext, uint32_t input_index, ModelInputTensorDesc* desc);
std::vector<float> run_image_model(void* qcontext, const std::vector<uint8_t>& input_data);
std::vector<float> run_text_model(void* qcontext, const std::vector<int64_t>& input_ids);
int destroy_network(void *qcontext);

#endif // MODEL_INVOKE_H

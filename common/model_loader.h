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

#ifndef _AMLNN_MODEL_LOADER_H_
#define _AMLNN_MODEL_LOADER_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <string>
#include "nnsdk2.h"

int init_network(std::string model_path, void*& qcontext);
int uninit_network(void* qcontext);
amlnn_tensor_attr query_input_attr(void* context, uint32_t index);
amlnn_tensor_attr query_output_attr(void *context, uint32_t index);
bool run_network(void* context, void* input_data, size_t input_size, std::vector<amlnn_output>& outputs);
#endif
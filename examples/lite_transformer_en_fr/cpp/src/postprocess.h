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

#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "nnsdk2.h"

bool run_multi_input_network(void *context,
                             const std::vector<void *> &input_data,
                             const std::vector<size_t> &input_sizes,
                             std::vector<amlnn_output> &outputs);

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);
int64_t greedy_next_token(const float *logits, int step, int vocabulary_size);

#endif
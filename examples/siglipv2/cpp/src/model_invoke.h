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

#ifndef MODEL_INVOKE_H
#define MODEL_INVOKE_H

#include <cstdint>
#include <vector>

std::vector<float> run_image_model(void *context, const std::vector<float> &input_data);
std::vector<float> run_text_model(void *context, const std::vector<int64_t> &input_ids);
#endif // MODEL_INVOKE_H
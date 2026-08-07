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

#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <string>
#include <vector>

std::vector<float> preprocess_image(const std::string &image_path);
std::vector<float> l2_normalize(const std::vector<float> &values);
std::vector<float> softmax(const std::vector<float> &logits);
float compute_similarity(const std::vector<float> &a, const std::vector<float> &b, float scale);

#endif
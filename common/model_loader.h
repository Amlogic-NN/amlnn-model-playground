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
#include "nn_sdk.h"

void* init_network(const char* model_path);
int uninit_network(void* qcontext);
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);
void* run_network(void* qcontext, std::vector<std::tuple<cv::Mat, float, std::tuple<int, int>>> input_tuples);

#endif
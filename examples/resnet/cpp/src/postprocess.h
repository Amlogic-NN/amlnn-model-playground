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
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "nnsdk2.h"

// Helper function to extract meaningful dimensions (ignores batch dim 1)
std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);

// Loads image, applies letterbox, normalizes, and quantizes for NPU
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

// Quantize float32 image to int8 for model input
cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point, int tensor_type);

// Sorting logits and printing Top-K classes
void postprocess_topk(float* logits,
                      int size,
                      const std::vector<std::string>& labels,
                      int k = 5);

// Load labels from a text file
std::vector<std::string> load_labels(const std::string& path);

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

#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

#define REC_MODEL_INPUT_WIDTH 320
#define REC_MODEL_INPUT_HEIGHT 48

// Normalization Constants
const float NORM_MEAN = 127.5f;
const float NORM_SCALE = 128.0f;

struct RecResult {
    std::string text;
    float score;
};

// Function declarations
std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);
std::vector<std::string> load_dict(const std::string& path);
std::vector<int16_t> quantize_input(const cv::Mat& float_img, const amlnn_tensor_attr& attr);
cv::Mat preprocess(const cv::Mat& image, const int dest_width, const int dest_height);
std::string postprocess_rec(float* out_data, const std::vector<int>& out_shape, const std::vector<std::string>& char_dict);

#endif // POSTPROCESS_H
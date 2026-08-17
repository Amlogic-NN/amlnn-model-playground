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

#include <cstdint>
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);

cv::Mat preprocess(cv::Mat img, std::tuple<int, int> new_shape);

std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr);

cv::Mat postprocess(
    float *output_ptr, const std::vector<int> &output_shape,
    int original_h, int original_w);

cv::Mat colorize_disparity(const cv::Mat &disparity);

#endif
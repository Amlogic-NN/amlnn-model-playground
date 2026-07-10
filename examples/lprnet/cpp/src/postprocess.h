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

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

// LPRNet Constants
const int LPR_MODEL_WIDTH = 94;
const int LPR_MODEL_HEIGHT = 24;

extern const std::vector<std::string> LPR_CHARS;
const int BLANK_IDX = 67;

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);

cv::Mat preprocess(const cv::Mat &image, const int dest_width, const int dest_height);

std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr);

std::pair<std::string, float> postprocess_lpr(float *out_data, const std::vector<int> &out_shape);

cv::Mat draw_detections(const cv::Mat& image, const std::string& text, float score);

#endif // POSTPROCESS_H
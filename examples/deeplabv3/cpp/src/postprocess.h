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

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <tuple>
#include "nnsdk2.h"

// Standard Pascal VOC Classes
extern const std::vector<std::string> VOC_CLASSES;
extern const std::vector<cv::Scalar> VOC_COLORS;

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);

std::tuple<cv::Mat, float, int, int, int, int> preprocess(const cv::Mat &img, int target_w, int target_h);

// Your robust helper implementation
std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr);

// Postprocesses the logits into a resized 8-bit mask
cv::Mat postprocess(float *out_data, const std::vector<int> &out_shape, int orig_w, int orig_h, int pad_left, int pad_top, int new_w, int new_h);

// Blends the integer mask with the original image
cv::Mat draw_segmentation(const cv::Mat &image, const cv::Mat &mask);

#endif // POSTPROCESS_H
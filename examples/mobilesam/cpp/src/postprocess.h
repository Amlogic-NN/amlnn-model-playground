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

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

struct ImageMeta
{
    int original_height;
    int original_width;
    int resized_height;
    int resized_width;
    int input_height;
    int input_width;
    float scale_x;
    float scale_y;
};

struct PromptPoint
{
    float x;
    float y;
    float label;
};

struct Prompt
{
    std::vector<PromptPoint> points;
    bool has_box = false;
    cv::Rect2f box;
};

struct MaskResult
{
    cv::Mat mask;
    float score = 0.0f;
    int index = -1;
};

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);
int get_tensor_element_count(const amlnn_tensor_attr &attr);
cv::Mat preprocess(const cv::Mat &image, std::tuple<int, int> new_shape, ImageMeta &meta);
std::vector<uint8_t> prepare_tensor(const float *data, int total_elements, const amlnn_tensor_attr &attr);
std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr);
bool build_prompt(const std::string &type, const std::string &values, const ImageMeta &meta,
                  std::vector<float> &point_coords, std::vector<float> &point_labels, Prompt &prompt);
MaskResult postprocess(float *mask_data, const amlnn_tensor_attr &mask_attr, float *score_data,
                       int score_elements, const ImageMeta &meta);
cv::Mat draw_result(const cv::Mat &image, const cv::Mat &mask, const Prompt &prompt);

#endif
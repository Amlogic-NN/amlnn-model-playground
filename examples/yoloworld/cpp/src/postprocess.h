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

#include <vector>
#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

extern const std::string WORLD_CLASSES[13];

struct OutputLayer
{
    float *buf;
    std::vector<int> shape;
    int area;
};


struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};
std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point, int tensor_type);

std::vector<Detection> postprocess(
    const std::vector<float*>& out_buffers,
    const std::vector<std::vector<int>>& out_shapes,
    std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
    float conf_thresh,
    float iou_threshold
);

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections);
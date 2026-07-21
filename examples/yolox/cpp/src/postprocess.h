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
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

constexpr int NUM_CLASSES = 80;

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(
    cv::Mat img, std::tuple<int, int> new_shape
);
std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr);
std::vector<Detection> postprocess(
    const std::vector<float *> &out_ptrs,
    const std::vector<std::vector<int>> &out_shapes,
    int input_h, int input_w,
    std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
    float conf_thresh, float iou_threshold
);
cv::Mat draw_detections(cv::Mat image, const std::vector<Detection> &detections);

#endif // POSTPROCESS_H
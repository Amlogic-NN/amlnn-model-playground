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
#include <tuple>
#include "nnsdk2.h"

// Structure to store Gesture detection results
struct GestureDetection
{
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

// Helper function to extract meaningful dimensions (ignores batch dim 1)
std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);

// Resizes and converts BGR to RGB, scaling values to [0, 1].
std::tuple<cv::Mat, int, int> preprocess(const cv::Mat &img, int input_size);

// Decodes 3 raw output tensors into bounding boxes and scores
std::vector<GestureDetection> postprocess(
    const std::vector<float *> &out_data,
    const std::vector<std::vector<int>> &out_shapes,
    const std::tuple<cv::Mat, int, int> &prep_info,
    float conf_thresh,
    float nms_thresh);

// Draws bounding boxes, scores, and class names
cv::Mat draw_detections(const cv::Mat &bgr, const std::vector<GestureDetection> &detections);
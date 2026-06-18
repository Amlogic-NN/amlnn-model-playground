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

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>
#include "nnsdk2.h"

#include "anchors.h"

#define NUM_COORDS 12

// BlazePoseDetection result structure
struct BlazePoseDetection
{
    float coords[NUM_COORDS + 1]; // 12 coords + 1 score
};

// COCO class names (80 classes)
extern const char *COCO_CLASSES[80];

// Helper function to extract meaningful dimensions (ignores batch dim 1)
std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);

// Preprocess image with letterbox resizing
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

// Quantize float32 image to int8 for model input
cv::Mat quantize_input(const cv::Mat &float_img, float scale, int32_t zero_point);

// Postprocess blazepose_detect outputs with DFL decoding
std::vector<BlazePoseDetection> postprocess(float *raw_boxes, float *raw_scores,
                                            std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                            float conf_threshold, float iou_threshold);

// Draw detections on image
cv::Mat draw_detections(cv::Mat image, const std::vector<BlazePoseDetection> &detections);


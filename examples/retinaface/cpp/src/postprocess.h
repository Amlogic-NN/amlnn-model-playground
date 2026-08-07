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

#include <vector>
#include <array>
#include <tuple>
#include <opencv2/opencv.hpp>

// Define the standard output structure for a detected face
struct FaceDetection
{
    float x1, y1, x2, y2;
    float score;
    std::array<float, 10> landmarks;
};

// Returns the expected number of priors for tensor size checking
int get_num_priors(int target_w, int target_h);

std::tuple<cv::Mat, float, int, int> preprocess(const cv::Mat &img, int target_w, int target_h);
cv::Mat quantize_input(const cv::Mat &float_img, float scale, int32_t zero_point, int tensor_type);

// Postprocessing logic
std::vector<FaceDetection> postprocess(float *loc, bool loc_planar,
                                       float *conf, bool conf_planar,
                                       float *landm, bool landm_planar,
                                       std::tuple<cv::Mat, float, int, int> input_tuple,
                                       int target_w, int target_h,
                                       float conf_thresh, float nms_thresh);

// Drawing logic
cv::Mat draw_detections(cv::Mat image, const std::vector<FaceDetection> &detections);
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
#include <tuple>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

struct Detection {
    float x1, y1, x2, y2;
    float score;
    std::vector<std::pair<float, float>> keypoints;
    std::vector<float> kpt_conf;
};

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);

// Preprocess
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

// Quantize
cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point);

// Postprocess handles the 4 outputs for Pose
std::vector<Detection> postprocess(float* bbox_data, const std::vector<int>& bbox_shape,
                                   float* score_data, const std::vector<int>& score_shape,
                                   float* kpt_conf_data, const std::vector<int>& kpt_conf_shape,
                                   float* kpt_xy_data, const std::vector<int>& kpt_xy_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold);

// Drawing detections with skeletons and keypoints
cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections);

#endif // POSTPROCESS_H
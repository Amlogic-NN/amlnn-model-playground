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
    int class_id;
    std::vector<float> mask_coeff;
};

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);
// Preprocess remains the same
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point);

// Added mask_coeff_data and shape to postprocess
std::vector<Detection> postprocess(float* bbox_data, const std::vector<int>& bbox_shape,
                                   float* score_data, const std::vector<int>& score_shape,
                                   float* mask_coeff_data, const std::vector<int>& mask_coeff_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold);

// Added proto_mask data, shape, and original scaling parameters for mask drawing
cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections,
                        float* proto_mask_data, const std::vector<int>& proto_shape,
                        float scale, std::tuple<int, int> pad);

#endif // POSTPROCESS_H
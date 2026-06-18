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
    std::vector<cv::Point2f> corners; // 4 corners of the Oriented Bounding Box
    float aabb_x1, aabb_y1, aabb_x2, aabb_y2; // Axis-Aligned box for NMS
    float score;
    int class_id;
};

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);
// Preprocess
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

// Quantize
cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point);

// Postprocess handles the 3 outputs for OBB (Bbox, Score, Angle)
std::vector<Detection> postprocess(float* bbox_data, const std::vector<int>& bbox_shape,
                                   float* score_data, const std::vector<int>& score_shape,
                                   float* angle_data, const std::vector<int>& angle_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold);

// Drawing detections with Oriented Bounding Boxes
cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections);

#endif // POSTPROCESS_H
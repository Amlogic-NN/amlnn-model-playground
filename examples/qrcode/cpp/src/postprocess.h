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
#include <string>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include "nnsdk2.h"

struct Detection
{
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    std::string text;
    std::vector<cv::Point> points;
};

// Helper function to extract meaningful dimensions (ignores batch dim 1)
std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);

// Preprocess: Resizes the image to target size, normalizes to 0-1, and returns scales
std::tuple<cv::Mat, float, float> preprocess(const cv::Mat &img, int input_width, int input_height);

// Postprocess: Parses model output, applies NMS, scales back to original image and applies padding
std::vector<Detection> postprocess(float *out_data, const std::vector<int> &out_shape,
                                          float sx, float sy, int orig_w, int orig_h,
                                          float conf_thresh, float iou_thresh, int pad);

// Decodes QR codes from the padded regions
std::vector<Detection> decode(const cv::Mat &orig_img, const std::vector<Detection> &detections);

// Draws bounding boxes, scores, and text on the image
cv::Mat draw_results(const cv::Mat &image, const std::vector<Detection> &detections);
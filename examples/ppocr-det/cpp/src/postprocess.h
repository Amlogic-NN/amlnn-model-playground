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

#ifndef __AMLNN_PADDLEOCR_DET_DEMO_POSTPROCESS_H__
#define __AMLNN_PADDLEOCR_DET_DEMO_POSTPROCESS_H__

#include <opencv2/opencv.hpp>
#include <vector>

#include "nn_sdk.h"

const int MIN_SIZE       = 3;
const int MAX_CANDIDATES = 1000;
const int MODEL_INPUT_WIDTH  = 640;
const int MODEL_INPUT_HEIGHT = 640;
const int MODEL_INPUT_CHANNELS = 3;
const float BOX_SCORE_THRESH = 0.5;
const float BOX_THRESH = 0.3;

typedef struct {
    float score;

    std::vector<cv::Point> box;
} Object;

void* run_paddleocr_network(void* qcontext, cv::Mat& image, const int width, const int height, const int channel);

int preprocess(const cv::Mat& image, cv::Mat& pre_image, const int width, const int height, float& scale );

int postprocess(float* out, const cv::Mat& image, float box_score_thresh, float box_thresh, std::vector<Object>& result, float scale);

std::vector<Object> find_box(const cv::Mat pred_map, const cv::Mat& bit_map,
                             const float box_score_thresh, const float unclip_ratio,
                             const cv::Mat& image, float scale);

std::vector<cv::Point> get_min_boxes(const std::vector<cv::Point>& in_vec, 
                        float& min_side_len, float& perimeter);

float get_box_score_fast(const cv::Mat& in_mat, const std::vector<cv::Point>& in_box);

std::vector<cv::Point> unclip(const std::vector<cv::Point>& in_box, float perimeter, float unclip_ratio);

bool cv_point_compare(const cv::Point& a, const cv::Point& b);

cv::Mat draw_objects(cv::Mat image, const std::vector<Object>& results);


#endif
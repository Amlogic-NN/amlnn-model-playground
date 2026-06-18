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

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

// ==========================================
// Model Parameters
// ==========================================
// DET requires exactly 640x640 based on the adla output size requirements
constexpr int DET_MODEL_WIDTH = 640;
constexpr int DET_MODEL_HEIGHT = 640;
constexpr int DET_MODEL_CHANNELS = 3;

constexpr int REC_MODEL_WIDTH = 320;
constexpr int REC_MODEL_HEIGHT = 48;
constexpr int REC_MODEL_CHANNELS = 3;

// Normalization Constants for REC
constexpr float NORM_MEAN = 127.5f;
constexpr float NORM_SCALE = 128.0f;

// Det Thresholds
constexpr float BOX_SCORE_THRESH = 0.5f;
constexpr float BOX_THRESH = 0.3f;
constexpr float UNCLIP_RATIO = 1.5f;
constexpr int MIN_SIZE = 3;
constexpr int MAX_CANDIDATES = 1000;

// ==========================================
// Structures
// ==========================================
struct Object
{
    std::vector<cv::Point> box; // 4 points of the text box
    float score;                // Detection score
    std::string text;           // Recognized text
    float rec_score;            // Recognition confidence
};

struct RecResult
{
    std::string text;
    float score;
};

// ==========================================
// Utils & Headers
// ==========================================
std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);
std::vector<std::string> load_dict(const std::string &path);

// Det Functions
std::tuple<cv::Mat, float> preprocess_det(const cv::Mat &image, const int width, const int height);
std::vector<int8_t> quantize_input_det(const cv::Mat &float_img, const amlnn_tensor_attr &attr);
std::vector<Object> postprocess_det(float *out, const std::vector<int> &shape, const cv::Mat &image, float box_score_thresh, float box_thresh, float scale);
std::vector<Object> find_box(const cv::Mat pred_map, const cv::Mat &bit_map, const float box_score_thresh, const float unclip_ratio, const cv::Mat &image, float scale);
std::vector<cv::Point> get_min_boxes(const std::vector<cv::Point> &in_vec, float &min_side_len, float &perimeter);
float get_box_score_fast(const cv::Mat &in_mat, const std::vector<cv::Point> &in_box);
std::vector<cv::Point> unclip(const std::vector<cv::Point> &in_box, float perimeter, float unclip_ratio);
bool cv_point_compare(const cv::Point &a, const cv::Point &b);

// Rec Functions
cv::Mat preprocess_rec(const cv::Mat &image, const int dest_width, const int dest_height);
std::vector<int16_t> quantize_input_rec(const cv::Mat &float_img, const amlnn_tensor_attr &attr);
std::string postprocess_rec(float *out_data, const std::vector<int> &out_shape, const std::vector<std::string> &char_dict);

// Draw
cv::Mat draw_ocr_results(cv::Mat image, const std::vector<Object> &results);

#endif // POSTPROCESS_H
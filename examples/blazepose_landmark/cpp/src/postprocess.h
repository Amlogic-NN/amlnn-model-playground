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
#include <string>
#include <tuple>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

const int NUM_LANDMARKS = 39;
const int LANDMARK_FEATURE_DIM = 5;
const int IMAGE_SIZE = 256;

// Struct for ROI
struct ROI {
    float x_center;
    float y_center;
    float box_size;
    float rotation;
};

// Struct for Landmarks
struct BlazePoseLandmark {
    std::vector<std::vector<double>> landmarks;
};

// Function Declarations
std::vector<std::vector<float>> load_detections(const std::string& txt_path);

std::tuple<cv::Mat, ROI> preprocess(cv::Mat img, std::vector<std::vector<float>> &detections, std::tuple<int, int> new_shape);

cv::Mat quantize_input(const cv::Mat &float_img, const amlnn_tensor_attr& attr);

std::vector<BlazePoseLandmark> postprocess(float *raw_landmarks, float *raw_heatmap, const ROI &roi);

cv::Mat draw_landmarks(cv::Mat image, const std::vector<BlazePoseLandmark> &landmarks, float score_threshold);

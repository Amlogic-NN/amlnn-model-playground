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

#ifndef _AMLNN_BLAZEPOSE_LANDMARK_POSTPROCESS_H_
#define _AMLNN_BLAZEPOSE_LANDMARK_POSTPROCESS_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>

#include "model_loader.h"

#define NUM_LANDMARKS 33
#define LANDMARK_OUT_DIM 4
#define LANDMARK_FEATURE_DIM 5
#define IMAGE_SIZE 256
// BlazePoseLandmark result structure

struct BlazePoseLandmark
{
    std::vector<std::vector<double>> landmarks; // [N][x,y,z,v]
};

// Preprocess image with letterbox resizing
std::tuple<cv::Mat, cv::Mat> preprocess(cv::Mat img, std::vector<std::vector<float>> &detections, std::tuple<int, int> new_shape);

// Quantize float32 image to int8 for model input
cv::Mat quantize_input(const cv::Mat &float_img, float scale = 0.000030518509447574615f, int16_t zero_point = 0);

// Postprocess blazepose_landmark outputs with DFL decoding
std::vector<BlazePoseLandmark> postprocess(nn_output *outdata, const cv::Mat &affine);

// Draw detections on image
cv::Mat draw_landmarks(cv::Mat image, const std::vector<BlazePoseLandmark> &landmarks, float score_threshold = 0.5);

#endif // _AMLNN_BLAZEPOSE_LANDMARK_POSTPROCESS_H_

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

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"

const int NUM_DETECTION_COORDS = 12;
const int NUM_MODEL_LANDMARKS = 39;
const int NUM_POSE_LANDMARKS = 33;

struct Detection
{
    std::array<float, NUM_DETECTION_COORDS> coords{};
    float score = 0.0f;
};

struct Roi
{
    float center_x = 0.0f;
    float center_y = 0.0f;
    float size = 0.0f;
    float rotation = 0.0f;
};

struct Landmark
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float visibility = 0.0f;
    float presence = 0.0f;
    cv::Point3f world{};
};

struct PoseResult
{
    float score = 0.0f;
    std::array<Landmark, NUM_POSE_LANDMARKS> landmarks{};
};

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);
std::vector<Detection> load_detections(const std::string &path);
Roi detection_to_roi(const Detection &detection, int image_width, int image_height);
cv::Mat preprocess(const cv::Mat &image, const Roi &roi, std::tuple<int, int> new_shape);
std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr);

bool postprocess(const std::vector<float *> &out_ptrs,
                 const std::vector<std::vector<int>> &out_shapes,
                 const Roi &roi, int image_width, int image_height,
                 float presence_threshold, PoseResult &result);

bool save_landmarks(const std::string &path, const std::vector<PoseResult> &results);

cv::Mat draw_detections(const cv::Mat &image,
                        const std::vector<PoseResult> &results,
                        float visibility_threshold = 0.5f);

#endif
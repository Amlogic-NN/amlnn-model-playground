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

#ifndef RETINAFACE_POSTPROCESS_H
#define RETINAFACE_POSTPROCESS_H

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

static constexpr int kInputW = 320;
static constexpr int kInputH = 320;

std::vector<std::array<float, 4>> generate_priors();

std::array<float, 10> decode_landm(const float* lm, int idx, int total, bool is_planar, const std::array<float, 4>& p);

std::array<float, 4> decode_box(const float* loc, int idx, int total, bool is_planar, const std::array<float, 4>& p);

float iou(const std::array<float, 4>& a, const std::array<float, 4>& b);

std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes, const std::vector<float>& scores, float thresh);

#endif
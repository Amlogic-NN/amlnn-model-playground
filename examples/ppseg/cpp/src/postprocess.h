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

// Helper function to extract meaningful dimensions (ignores batch dim 1)
std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr);

// Resize and Normalizes image to [0, 1] then applies Mean/Std
cv::Mat preprocess(cv::Mat img, int input_height, int input_width);

std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr);

// Extracts the 19-class argmax output and resizes it back to original image dimensions
cv::Mat postprocess(float* out_data, const std::vector<int>& out_shape, int orig_w, int orig_h);

// Applies Cityscapes ColorMap and Alpha blends onto the original image
cv::Mat draw_segmentation(cv::Mat image, const cv::Mat& pred_mask, float alpha);

#endif // POSTPROCESS_H
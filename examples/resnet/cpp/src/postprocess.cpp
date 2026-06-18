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

#include "postprocess.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>

#define LOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
        {
            shape.push_back(attr.dims[i]);
        }
    }
    return shape;
}


std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape) {
    if (img.empty()) {
        LOGE("Preprocess received empty image");
        return {};
    }

    cv::Mat img_rgb;
    // 1. Convert to RGB
    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);

    // 2. Direct Resize
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    // 3. Convert to float [0.0, 1.0]
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

    // 4. Standard ImageNet Normalization
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;
    cv::merge(channels, img_float);

    // Return 1.0 scale and 0 padding since we used direct resize
    return std::make_tuple(img_float, 1.0f, std::make_tuple(0, 0));
}

// Ensure you pass tensor_type here from main!
cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point, int tensor_type) {
    if (float_img.empty() || float_img.type() != CV_32FC3) {
        LOGE("quantize_input: Invalid input image (must be CV_32FC3)");
        return cv::Mat();
    }

    int total_elements = float_img.total() * float_img.channels();
    const float* src_ptr = (const float*)float_img.data;

    if (tensor_type == 3) {
        // UINT8 HANDLING
        cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8UC3);
        uint8_t* dst_ptr = (uint8_t*)quantized_img.data;

        for (int i = 0; i < total_elements; ++i) {
            float val = std::round(src_ptr[i] / scale) + zero_point;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
        return quantized_img;
    }
    else {
        // INT8 HANDLING
        cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8SC3);
        int8_t* dst_ptr = (int8_t*)quantized_img.data;

        for (int i = 0; i < total_elements; ++i) {
            float val = std::round(src_ptr[i] / scale) + zero_point;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }
        return quantized_img;
    }
}

void postprocess_topk(float* logits, int size, const std::vector<std::string>& labels, int k) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    std::cout << "\nTop-" << k << " Results:" << std::endl;
    for (int i = 0; i < k; ++i) {
        int idx = indices[i];
        std::string name = (idx < (int)labels.size()) ? labels[idx] : "N/A";
        printf("%d: %-20s  score=%.6f\n", i + 1, name.c_str(), logits[idx]);
    }
}

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: Could not open label file: " << path << std::endl;
        return labels;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}
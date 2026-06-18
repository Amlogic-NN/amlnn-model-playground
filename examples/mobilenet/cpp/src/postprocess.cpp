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

#include "postprocess.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>

#define LOGI(...)            \
    do                       \
    {                        \
        printf(__VA_ARGS__); \
        printf("\n");        \
    } while (0)
#define LOGE(...)                     \
    do                                \
    {                                 \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

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

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape)
{
    if (img.empty())
    {
        LOGE("Preprocess received empty image");
        return {};
    }

    // 1. Convert BGR to RGB
    cv::Mat img_rgb;
    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);

    // 2. Direct resize
    cv::Mat resized_img;
    cv::resize(img_rgb, resized_img, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat img_float;
    resized_img.convertTo(img_float, CV_32FC3);

    return std::make_tuple(img_float, 1.0f, std::make_tuple(0, 0));
}

cv::Mat quantize_input(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
{
    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        LOGE("quantize_input: Invalid input image");
        return cv::Mat();
    }

    int total_elements = float_img.total() * float_img.channels();
    const float *src_ptr = (const float *)float_img.data;

    if (attr.type == 2)
    {
        // INT8 MODE
        cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8SC3);
        int8_t *dst_ptr = (int8_t *)quantized_img.data;

        for (int i = 0; i < total_elements; ++i)
        {
            float pixel_val = (src_ptr[i] / 127.5f) - 1.0f; // [0, 255] -> [-1.0, 1.0]
            float q_val = std::round(pixel_val / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, q_val)));
        }
        return quantized_img;
    }
    else if (attr.type == 3)
    {
        // UINT8 MODE
        cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8UC3);
        uint8_t *dst_ptr = (uint8_t *)quantized_img.data;

        for (int i = 0; i < total_elements; ++i)
        {
            float pixel_val = (src_ptr[i] / 127.5f) - 1.0f; // [0, 255] -> [-1.0, 1.0]
            float q_val = std::round(pixel_val / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, q_val)));
        }
        return quantized_img;
    }
    else
    {
        LOGE("quantize_input: Unsupported tensor_type %d", attr.type);
        return cv::Mat();
    }
}

static void softmax(float *data, int size)
{
    float max_val = data[0];
    for (int i = 1; i < size; ++i)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    for (int i = 0; i < size; ++i)
    {
        data[i] /= sum;
    }
}

void postprocess_topk(float *buf, int size, const std::vector<std::string> &labels, int k)
{
    softmax(buf, size);
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort to get Top-K
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b)
                      { return buf[a] > buf[b]; });

    std::cout << "\n    Top-" << k << " Results:" << std::endl;
    for (int i = 0; i < k; ++i)
    {
        int idx = indices[i];
        std::string name = (idx < (int)labels.size()) ? labels[idx] : "Unknown(" + std::to_string(idx) + ")";
        printf("      %d. %-20s  prob=%.6f\n", i + 1, name.c_str(), buf[idx]);
    }
}

std::vector<std::string> load_labels(const std::string &path)
{
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "Warning: Could not open label file: " << path << std::endl;
        return labels;
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (!line.empty())
            labels.push_back(line);
    }
    return labels;
}
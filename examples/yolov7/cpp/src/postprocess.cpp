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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
            shape.push_back(attr.dims[i]);
    }
    return shape;
}

cv::Mat preprocess(cv::Mat img, std::tuple<int, int> new_shape)
{
    if (img.empty())
        return {};

    cv::Mat img_rgb;
    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);

    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32F);
    return img_float;
}

std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;
    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        std::cerr << "prepare_input_tensor: Invalid input image" << std::endl;
        return tensor_data;
    }

    int total_elements = static_cast<int>(float_img.total() * float_img.channels());
    const float *src_ptr = float_img.ptr<float>();

    // CREStereo uses RGB values in [0, 255] with no normalization.
    if (attr.type == AMLNN_TENSOR_FLOAT32)
    {
        tensor_data.resize(total_elements * sizeof(float));
        std::memcpy(tensor_data.data(), float_img.data, tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(total_elements * sizeof(int16_t));
        int16_t *dst_ptr = reinterpret_cast<int16_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, value)));
        }
    }
    else
    {
        std::cerr << "prepare_input_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

cv::Mat postprocess(
    float *output_ptr, const std::vector<int> &output_shape,
    int original_h, int original_w)
{
    if (output_ptr == nullptr || output_shape.size() != 3)
    {
        std::cerr << "Unexpected CREStereo output shape." << std::endl;
        return {};
    }

    int model_h = output_shape[0];
    int model_w = output_shape[1];
    int channels = output_shape[2];
    if (channels != 2)
    {
        std::cerr << "Expected CREStereo output channels = 2, got " << channels << std::endl;
        return {};
    }

    // Output channel 0 is horizontal disparity.
    cv::Mat disparity(model_h, model_w, CV_32F);
    for (int y = 0; y < model_h; ++y)
    {
        float *dst = disparity.ptr<float>(y);
        for (int x = 0; x < model_w; ++x)
        {
            float value = output_ptr[(y * model_w + x) * channels];
            dst[x] = std::isfinite(value) ? value : 0.0f;
        }
    }

    cv::Mat resized_disparity;
    cv::resize(disparity, resized_disparity, cv::Size(original_w, original_h), 0, 0, cv::INTER_LINEAR);

    // Disparity is measured in pixels, so scale values with the image-width ratio.
    resized_disparity *= static_cast<float>(original_w) / static_cast<float>(model_w);
    return resized_disparity;
}

cv::Mat colorize_disparity(const cv::Mat &disparity)
{
    if (disparity.empty() || disparity.type() != CV_32F)
        return {};

    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(disparity, &min_value, &max_value);

    cv::Mat normalized;
    if (max_value <= min_value)
    {
        normalized = cv::Mat::zeros(disparity.size(), CV_8U);
    }
    else
    {
        double scale = 255.0 / (max_value - min_value);
        disparity.convertTo(normalized, CV_8U, scale, -min_value * scale);
    }

    cv::Mat color_disparity;
    cv::applyColorMap(normalized, color_disparity, cv::COLORMAP_MAGMA);
    return color_disparity;
}
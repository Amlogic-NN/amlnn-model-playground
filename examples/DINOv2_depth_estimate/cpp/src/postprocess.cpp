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
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

const cv::Scalar IMAGENET_MEAN(123.675f, 116.28f, 103.53f);
const cv::Scalar IMAGENET_STD(58.395f, 57.12f, 57.375f);

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr)
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

cv::Mat preprocess(cv::Mat img, std::tuple<int, int> new_shape)
{
    cv::Mat img_rgb;
    if (img.empty())
        return {};

    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGRA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int input_height = std::get<0>(new_shape);
    int input_width = std::get<1>(new_shape);

    cv::Mat resized_img;
    cv::resize(img_rgb, resized_img, cv::Size(input_width, input_height), 0, 0, cv::INTER_CUBIC);

    cv::Mat img_float;
    resized_img.convertTo(img_float, CV_32FC3);
    cv::subtract(img_float, IMAGENET_MEAN, img_float);
    cv::divide(img_float, IMAGENET_STD, img_float);

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

    int total_elements = float_img.total() * float_img.channels();
    const float *src_ptr = float_img.ptr<float>();

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
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
    }
    else
    {
        std::cerr << "prepare_input_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

std::vector<uint8_t> prepare_feature_tensor(const std::vector<float> &features, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;

    if (attr.type == AMLNN_TENSOR_FLOAT32)
    {
        tensor_data.resize(features.size() * sizeof(float));
        std::memcpy(tensor_data.data(), features.data(), tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(features.size() * sizeof(int16_t));
        int16_t *dst_ptr = reinterpret_cast<int16_t *>(tensor_data.data());
        for (size_t i = 0; i < features.size(); ++i)
        {
            float val = std::round(features[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(features.size() * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());
        for (size_t i = 0; i < features.size(); ++i)
        {
            float val = std::round(features[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(features.size() * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());
        for (size_t i = 0; i < features.size(); ++i)
        {
            float val = std::round(features[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
    }
    else
    {
        std::cerr << "prepare_feature_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

std::vector<float> concatenate_backbone_outputs(
    const std::vector<amlnn_output> &outputs,
    const std::vector<std::vector<int>> &output_shapes,
    int expected_size)
{
    if (outputs.size() != NUM_BACKBONE_OUTPUTS || output_shapes.size() != NUM_BACKBONE_OUTPUTS)
    {
        std::cerr << "Expected " << NUM_BACKBONE_OUTPUTS << " backbone outputs." << std::endl;
        return {};
    }

    if (output_shapes[0].size() != 2)
    {
        std::cerr << "Unexpected backbone output shape." << std::endl;
        return {};
    }

    int token_count = output_shapes[0][0];
    int embed_dim = output_shapes[0][1];
    int output_size = token_count * embed_dim;
    std::vector<float> concat_features(NUM_BACKBONE_OUTPUTS * output_size);

    for (int output_index = 0; output_index < NUM_BACKBONE_OUTPUTS; ++output_index)
    {
        if (output_shapes[output_index].size() != 2 ||
            output_shapes[output_index][0] != token_count ||
            output_shapes[output_index][1] != embed_dim)
        {
            std::cerr << "Backbone output shapes do not match." << std::endl;
            return {};
        }

        float *output_ptr = reinterpret_cast<float *>(outputs[output_index].buf);
        float *dst_ptr = concat_features.data() + output_index * output_size;
        std::memcpy(dst_ptr, output_ptr, output_size * sizeof(float));
    }

    if (concat_features.size() != static_cast<size_t>(expected_size))
    {
        std::cerr << "Unexpected concatenated feature size: " << concat_features.size()
                  << ", expected: " << expected_size << std::endl;
        return {};
    }

    return concat_features;
}

cv::Mat postprocess(
    float *output,
    int output_size,
    const std::vector<int> &output_shape,
    const cv::Size &original_size,
    float min_depth,
    float max_depth)
{
    if (output_shape.size() != 2)
    {
        std::cerr << "Unexpected depth output shape." << std::endl;
        return {};
    }

    int output_height = output_shape[0];
    int output_width = output_shape[1];

    if (output_height * output_width != output_size)
    {
        std::cerr << "Unexpected depth output size: " << output_size << std::endl;
        return {};
    }

    cv::Mat depth_map(output_height, output_width, CV_32FC1, output);
    depth_map = depth_map.clone();
    cv::resize(depth_map, depth_map, original_size, 0, 0, cv::INTER_LINEAR);

    for (int y = 0; y < depth_map.rows; ++y)
    {
        float *row_ptr = depth_map.ptr<float>(y);
        for (int x = 0; x < depth_map.cols; ++x)
        {
            if (std::isnan(row_ptr[x]) || row_ptr[x] == -std::numeric_limits<float>::infinity())
                row_ptr[x] = min_depth;
            else if (row_ptr[x] == std::numeric_limits<float>::infinity())
                row_ptr[x] = max_depth;

            row_ptr[x] = std::max(min_depth, std::min(max_depth, row_ptr[x]));
        }
    }

    return depth_map;
}

cv::Mat colorize_depth(const cv::Mat &depth_map)
{
    if (depth_map.empty() || depth_map.type() != CV_32FC1)
        return {};

    double depth_min;
    double depth_max;
    cv::minMaxLoc(depth_map, &depth_min, &depth_max);

    cv::Mat normalized;
    if (depth_max <= depth_min)
    {
        normalized = cv::Mat::zeros(depth_map.size(), CV_8UC1);
    }
    else
    {
        cv::Mat inverted = depth_max - depth_map;
        inverted.convertTo(normalized, CV_8UC1, 255.0 / (depth_max - depth_min));
    }

    cv::Mat colorized;
    cv::applyColorMap(normalized, colorized, cv::COLORMAP_INFERNO);
    return colorized;
}
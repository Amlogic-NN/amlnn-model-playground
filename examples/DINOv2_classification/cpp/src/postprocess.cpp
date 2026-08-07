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
#include <iomanip>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

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
    int original_height = img_rgb.rows;
    int original_width = img_rgb.cols;

    int resized_height;
    int resized_width;

    if (original_height < original_width)
    {
        resized_height = RESIZE_SHORT_SIDE;
        resized_width = static_cast<int>(std::round(static_cast<float>(original_width) * RESIZE_SHORT_SIDE / original_height));
    }
    else
    {
        resized_width = RESIZE_SHORT_SIDE;
        resized_height = static_cast<int>(std::round(static_cast<float>(original_height) * RESIZE_SHORT_SIDE / original_width));
    }

    cv::Mat resized_img;
    cv::resize(img_rgb, resized_img, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_CUBIC);

    int crop_x = (resized_width - input_width) / 2;
    int crop_y = (resized_height - input_height) / 2;
    cv::Mat cropped_img = resized_img(cv::Rect(crop_x, crop_y, input_width, input_height)).clone();

    cv::Mat img_float;
    cropped_img.convertTo(img_float, CV_32FC3);
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
    int concat_channels = NUM_BACKBONE_OUTPUTS * embed_dim;
    std::vector<float> concat_features(token_count * concat_channels);

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

        for (int token_index = 0; token_index < token_count; ++token_index)
        {
            float *src_ptr = output_ptr + token_index * embed_dim;
            float *dst_ptr = concat_features.data() + token_index * concat_channels + output_index * embed_dim;
            std::memcpy(dst_ptr, src_ptr, embed_dim * sizeof(float));
        }
    }

    if (concat_features.size() != static_cast<size_t>(expected_size))
    {
        std::cerr << "Unexpected concatenated feature size: " << concat_features.size()
                  << ", expected: " << expected_size << std::endl;
        return {};
    }

    return concat_features;
}

std::vector<std::string> load_class_names(const std::string &path)
{
    std::vector<std::string> class_names;
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cerr << "Warning: Could not load class names from '" << path << "'. Fallback to generic IDs." << std::endl;
        return class_names;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (!line.empty())
            class_names.push_back(line);
    }

    return class_names;
}

std::vector<std::pair<int, float>> postprocess(float *output, int output_size, int topk)
{
    std::vector<std::pair<int, float>> results;
    if (output == nullptr || output_size <= 0)
        return results;

    float max_logit = *std::max_element(output, output + output_size);
    std::vector<float> probabilities(output_size);
    float probability_sum = 0.0f;

    for (int i = 0; i < output_size; ++i)
    {
        probabilities[i] = std::exp(output[i] - max_logit);
        probability_sum += probabilities[i];
    }

    for (float &probability : probabilities)
        probability /= probability_sum;

    topk = std::min(topk, output_size);
    std::vector<int> indices(output_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(),
                      [&probabilities](int left, int right)
                      { return probabilities[left] > probabilities[right]; });

    for (int i = 0; i < topk; ++i)
    {
        int class_id = indices[i];
        results.push_back(std::make_pair(class_id, probabilities[class_id]));
    }

    return results;
}
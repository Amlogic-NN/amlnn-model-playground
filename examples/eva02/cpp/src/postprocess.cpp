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
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>

const float EVA02_MEAN[3] = {122.7709383f, 116.7460125f, 104.09373615f};
const float EVA02_STD[3] = {68.5005327f, 66.6321579f, 70.32316305f};

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
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int orig_h = img.rows;
    int orig_w = img.cols;
    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);

    float scale = std::max(static_cast<float>(target_h) / orig_h,
                           static_cast<float>(target_w) / orig_w);
    int new_h = static_cast<int>(round(orig_h * scale));
    int new_w = static_cast<int>(round(orig_w * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    int crop_top = (new_h - target_h) / 2;
    int crop_left = (new_w - target_w) / 2;
    cv::Mat img_cropped = img_resized(cv::Rect(crop_left, crop_top, target_w, target_h)).clone();

    cv::Mat img_float;
    img_cropped.convertTo(img_float, CV_32F);
    cv::subtract(img_float, cv::Scalar(EVA02_MEAN[0], EVA02_MEAN[1], EVA02_MEAN[2]), img_float);
    cv::divide(img_float, cv::Scalar(EVA02_STD[0], EVA02_STD[1], EVA02_STD[2]), img_float);

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

std::vector<std::string> load_class_names(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Warning: Could not load class names from '" << path << "'. Fallback to generic IDs." << std::endl;
        return {};
    }

    std::vector<std::string> class_names;
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

std::vector<ClassificationResult> postprocess(const std::vector<float *> &out_ptrs,
                                              const std::vector<std::vector<int>> &out_shapes,
                                              const std::vector<std::string> &class_names,
                                              int top_k)
{
    if (out_ptrs.empty() || out_shapes.empty())
        return {};

    int num_classes = 1;
    for (int dim : out_shapes[0])
        num_classes *= dim;

    const float *logits = out_ptrs[0];
    float max_logit = *std::max_element(logits, logits + num_classes);

    std::vector<float> probabilities(num_classes);
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i)
    {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    for (float &probability : probabilities)
        probability /= sum_exp;

    std::vector<int> indices(num_classes);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b)
              { return probabilities[a] > probabilities[b]; });

    int result_count = std::min(top_k, num_classes);
    std::vector<ClassificationResult> results;
    for (int i = 0; i < result_count; ++i)
    {
        int class_id = indices[i];
        std::string class_name = class_id < static_cast<int>(class_names.size())
                                     ? class_names[class_id]
                                     : "class_" + std::to_string(class_id);
        results.push_back({class_id, probabilities[class_id], class_name});
    }

    return results;
}
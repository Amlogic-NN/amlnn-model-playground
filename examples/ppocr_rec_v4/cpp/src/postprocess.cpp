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
#include <algorithm>
#include <fstream>
#include <cmath>

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
            shape.push_back(attr.dims[i]);
    }
    return shape;
}

std::vector<std::string> load_dict(const std::string &path)
{
    std::vector<std::string> dict;
    std::ifstream in(path);
    if (!in.is_open())
        return dict;
    std::string line;
    while (std::getline(in, line))
    {
        dict.push_back(line);
    }
    dict.push_back(" "); // Space character mapping
    return dict;
}

cv::Mat preprocess(const cv::Mat &image, const int dest_width, const int dest_height)
{
    if (image.empty())
        return cv::Mat();

    cv::Mat rgb_img;
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

    float ratio = (float)rgb_img.cols / (float)rgb_img.rows;
    int resize_w = std::min(int(dest_height * ratio), dest_width);

    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(resize_w, dest_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat float_resized;
    resized_img.convertTo(float_resized, CV_32FC3);

    // Normalize
    float_resized = (float_resized - cv::Scalar(NORM_MEAN, NORM_MEAN, NORM_MEAN)) / NORM_SCALE;

    // Calculate background padding value
    float pad_value = -NORM_MEAN / NORM_SCALE;

    cv::Mat pre_image(dest_height, dest_width, CV_32FC3, cv::Scalar(pad_value, pad_value, pad_value));
    cv::Rect roi_rect(0, 0, resize_w, dest_height);
    float_resized.copyTo(pre_image(roi_rect));

    return pre_image;
}

std::vector<int16_t> quantize_input(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
{
    std::vector<int16_t> quantized_data;

    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        std::cerr << "quantize_rec_tensor_int16: Invalid input image" << std::endl;
        return quantized_data;
    }

    int total_elements = float_img.total() * float_img.channels();
    quantized_data.resize(total_elements);

    const float *src_ptr = float_img.ptr<float>();
    float scale = attr.scale;
    int32_t zp = attr.zp;

    for (int i = 0; i < total_elements; ++i)
    {
        float val = std::round(src_ptr[i] / scale) + zp;
        quantized_data[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, val)));
    }

    return quantized_data;
}

std::string postprocess_rec(float *out_data, const std::vector<int> &out_shape, const std::vector<std::string> &char_dict)
{
    std::string result = "";
    if (out_data == nullptr || out_shape.size() < 2)
        return result;

    int seq_len = out_shape[out_shape.size() - 2];
    int num_classes = out_shape[out_shape.size() - 1];

    int blank_idx = 0;
    int pre_argmax_idx = -1;
    float total_score = 0.0f;
    int valid_char_count = 0;

    for (int t = 0; t < seq_len; ++t)
    {
        float raw_max_score = -1.0f;
        int argmax_idx = -1;

        // 1. Find the raw max score
        for (int c = 0; c < num_classes; ++c)
        {
            float val = out_data[t * num_classes + c];
            if (val > raw_max_score)
            {
                raw_max_score = val;
                argmax_idx = c;
            }
        }

        // 3. CTC Decoding logic
        if (argmax_idx != blank_idx && argmax_idx != pre_argmax_idx)
        {
            int char_idx = argmax_idx - 1;
            if (char_idx >= 0 && char_idx < char_dict.size())
            {
                result += char_dict[char_idx];
                valid_char_count++;
            }
        }
        pre_argmax_idx = argmax_idx;
    }

    return result;
}
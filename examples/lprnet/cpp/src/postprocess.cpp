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
#include <iomanip>
#include <sstream>

const std::vector<std::string> LPR_CHARS = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "0", "1", "2", "3", "4", "5", "6", "7", "8",
    "9", "A", "B", "C", "D", "E", "F", "G", "H", "J",
    "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z", "学", "警", "-"
};

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        // Ignore batch size or dummy dimensions of 1
        if (attr.dims[i] > 1)
            shape.push_back(attr.dims[i]);
    }
    return shape;
}

cv::Mat preprocess(const cv::Mat &image, const int dest_width, const int dest_height)
{
    if (image.empty())
        return cv::Mat();

    // 1. Direct Resize
    // The model uses BGR, so we don't convert to RGB.
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(dest_width, dest_height), 0, 0, cv::INTER_LINEAR);

    // 2. Convert to float and apply LPR Normalization
    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32FC3);

    float_img = (float_img - cv::Scalar(127.5f, 127.5f, 127.5f)) / 128.0f;

    return float_img;
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
    else if (attr.type == AMLNN_TENSOR_FLOAT16)
    {
        cv::Mat fp16_img;
        float_img.convertTo(fp16_img, CV_16FC3);
        cv::Mat flat_img = fp16_img.isContinuous() ? fp16_img : fp16_img.clone();

        tensor_data.resize(total_elements * sizeof(uint16_t));
        std::memcpy(tensor_data.data(), flat_img.data, tensor_data.size());
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

std::pair<std::string, float> postprocess_lpr(float *out_data, const std::vector<int> &out_shape)
{
    if (out_data == nullptr || out_shape.size() < 2)
        return {"", 0.0f};

    int seq_len = 0;
    int num_classes = 0;
    bool is_transposed = false;

    // Handle potential transpose: shape can be [18, 68] or [68, 18]
    if (out_shape[0] == LPR_CHARS.size()) {
        num_classes = out_shape[0];
        seq_len = out_shape[1];
        is_transposed = true; // Maps to shape [68, 18]
    } else {
        seq_len = out_shape[0];
        num_classes = out_shape[1]; // Maps to shape [18, 68]
    }

    std::string text = "";
    float total_score = 0.0f;
    int valid_count = 0;
    int pre_idx = -1;

    for (int i = 0; i < seq_len; ++i)
    {
        float max_score = -1e9f;
        int max_idx = -1;

        // Find argmax for current sequence step
        for (int c = 0; c < num_classes; ++c)
        {
            float score = is_transposed ? out_data[c * seq_len + i] : out_data[i * num_classes + c];
            if (score > max_score)
            {
                max_score = score;
                max_idx = c;
            }
        }

        // CTC greedy decode: ignore blanks and repeated tokens
        if (max_idx != BLANK_IDX && max_idx != pre_idx)
        {
            if (max_idx < LPR_CHARS.size())
            {
                text += LPR_CHARS[max_idx];
                total_score += max_score;
                valid_count++;
            }
        }
        pre_idx = max_idx;
    }

    float avg_score = (valid_count > 0) ? (total_score / valid_count) : 0.0f;
    return {text, avg_score};
}

cv::Mat draw_detections(const cv::Mat& image, const std::string& text, float score)
{
    // Pad image at the top
    cv::Mat padded_img;
    cv::copyMakeBorder(image, padded_img, 40, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    // Note: Standard OpenCV `putText` doesn't natively render Chinese characters well without `cv::freetype`.
    // It will render as '?' on most systems, but the file structure and console output will still be correct.
    std::ostringstream stream;
    stream << text << " (" << std::fixed << std::setprecision(2) << score << ")";

    cv::putText(padded_img, stream.str(), cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    return padded_img;
}
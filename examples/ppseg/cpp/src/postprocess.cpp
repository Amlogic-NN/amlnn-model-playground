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
#include <cmath>
#include <algorithm>

// Standard Cityscapes 19 Classes Colors in BGR format for OpenCV
const cv::Vec3b CITYSCAPES_COLORS[19] = {
    cv::Vec3b(128, 64, 128), cv::Vec3b(244, 35, 232), cv::Vec3b(70, 70, 70),
    cv::Vec3b(102, 102, 156), cv::Vec3b(190, 153, 153), cv::Vec3b(153, 153, 153),
    cv::Vec3b(250, 170, 30), cv::Vec3b(220, 220, 0), cv::Vec3b(107, 142, 35),
    cv::Vec3b(152, 251, 152), cv::Vec3b(70, 130, 180), cv::Vec3b(220, 20, 60),
    cv::Vec3b(255, 0, 0), cv::Vec3b(0, 0, 142), cv::Vec3b(0, 0, 70),
    cv::Vec3b(0, 60, 100), cv::Vec3b(0, 80, 100), cv::Vec3b(0, 0, 230),
    cv::Vec3b(119, 11, 32)
};

const int NUM_CLASS = 19;
const cv::Scalar MEAN(123.675f, 116.28f, 103.53f);
const cv::Scalar STD(58.395f, 57.12f, 57.375f);

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

cv::Mat preprocess(cv::Mat img, int input_height, int input_width) {
    cv::Mat img_rgb;
    if (img.empty())
        return {};

    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    // Direct resize
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32FC3);

    // Normalize
    img_float = (img_float - MEAN) / STD;

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

    // Quantize with the tensor scale/zero point and saturate to the target type.
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

cv::Mat postprocess(float* out_data, const std::vector<int>& out_shape, int orig_w, int orig_h)
{
    int h = out_shape[0];
    int w = out_shape[1];
    int c = NUM_CLASS;

    cv::Mat pred_mask(h, w, CV_8UC1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int best_class = 0;
            float max_score = -1e9f;

            for (int cls = 0; cls < c; ++cls) {
                int idx = (y * w + x) * c + cls;

                float score = out_data[idx];
                if (score > max_score) {
                    max_score = score;
                    best_class = cls;
                }
            }
            pred_mask.at<uchar>(y, x) = static_cast<uchar>(best_class);
        }
    }

    cv::Mat mask_resized;
    cv::resize(pred_mask, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

    return mask_resized;
}

cv::Mat draw_segmentation(cv::Mat image, const cv::Mat& pred_mask, float alpha) {
    cv::Mat drawn_image = image.clone();
    cv::Mat color_mask = cv::Mat::zeros(image.size(), CV_8UC3);

    for (int y = 0; y < pred_mask.rows; ++y) {
        for (int x = 0; x < pred_mask.cols; ++x) {
            uchar class_id = pred_mask.at<uchar>(y, x);
            if (class_id < 19) {
                color_mask.at<cv::Vec3b>(y, x) = CITYSCAPES_COLORS[class_id];
            }
        }
    }

    // Alpha blend color mask over the original image
    cv::addWeighted(drawn_image, 1.0f - alpha, color_mask, alpha, 0, drawn_image);
    return drawn_image;
}
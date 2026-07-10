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

const float MEAN[3] = {123.675f, 116.280f, 103.530f};
const float STD[3] = {58.395f, 57.120f, 57.375f};

const std::vector<std::string> VOC_CLASSES = {
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"};

const std::vector<cv::Scalar> VOC_COLORS = []()
{
    std::vector<cv::Scalar> colors(21);
    for (int i = 0; i < 21; i++)
    {
        int r = 0, g = 0, b = 0, id = i;
        for (int j = 0; j < 8; j++)
        {
            if (id & (1 << 0))
                r |= (1 << (7 - j));
            if (id & (1 << 1))
                g |= (1 << (7 - j));
            if (id & (1 << 2))
                b |= (1 << (7 - j));
            id >>= 3;
        }
        colors[i] = cv::Scalar(b, g, r);
    }
    return colors;
}();

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1 || attr.n_dims == 4)
        {
            shape.push_back(attr.dims[i]);
        }
    }
    return shape;
}

std::tuple<cv::Mat, float, int, int, int, int> preprocess(const cv::Mat &img, int target_w, int target_h)
{
    cv::Mat img_rgb;
    if (img.empty()) return {};

    if (img.channels() == 4) cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3) cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else img_rgb = img.clone();

    int orig_h = img.rows, orig_w = img.cols;
    float scale = std::min(static_cast<float>(target_w) / orig_w,
                           static_cast<float>(target_h) / orig_h);

    int new_w = static_cast<int>(std::round(orig_w * scale));
    int new_h = static_cast<int>(std::round(orig_h * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_w = target_w - new_w;
    int pad_h = target_h - new_h;
    int pad_left = static_cast<int>(std::round(pad_w / 2.0 - 0.1));
    int pad_right = static_cast<int>(std::round(pad_w / 2.0 + 0.1));
    int pad_top = static_cast<int>(std::round(pad_h / 2.0 - 0.1));
    int pad_bottom = static_cast<int>(std::round(pad_h / 2.0 + 0.1));

    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Create CV_32FC3 (NHWC) Mat
    cv::Mat float_img;
    img_padded.convertTo(float_img, CV_32FC3);

    // Apply Mean and Std directly to the NHWC Mat
    float* float_data = float_img.ptr<float>();
    int total_pixels = target_w * target_h;

    for (int i = 0; i < total_pixels; ++i)
    {
        float_data[i * 3 + 0] = (float_data[i * 3 + 0] - MEAN[0]) / STD[0];
        float_data[i * 3 + 1] = (float_data[i * 3 + 1] - MEAN[1]) / STD[1];
        float_data[i * 3 + 2] = (float_data[i * 3 + 2] - MEAN[2]) / STD[2];
    }

    return std::make_tuple(float_img, scale, pad_left, pad_top, new_w, new_h);
}

// Robust input prep
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

cv::Mat postprocess(float *out_data, const std::vector<int> &out_shape, int orig_w, int orig_h, int pad_left, int pad_top, int new_w, int new_h)
{
    int num_classes = 21;
    int height = 512;
    int width = 512;

    // 1. Auto-detect logical shape (NCHW vs NHWC) exposed by the API
    if (out_shape.size() >= 4) {
        if (out_shape[1] == 21) {
            // Logically NCHW [1, 21, 512, 512]
            num_classes = out_shape[1];
            height = out_shape[2];
            width = out_shape[3];
        } else if (out_shape[3] == 21) {
            // Logically NHWC [1, 512, 512, 21]
            height = out_shape[1];
            width = out_shape[2];
            num_classes = out_shape[3];
        }
    }

    cv::Mat mask_2d(height, width, CV_8UC1);

    // 2. Argmax over the 21 classes (Assuming physical NHWC layout in memory)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float max_val = -1e9f;
            int best_cls = 0;

            // Base index for NHWC: (y * width + x) * num_classes
            int base_idx = (y * width + x) * num_classes;

            for (int c = 0; c < num_classes; ++c)
            {
                float val = out_data[base_idx + c];
                if (val > max_val)
                {
                    max_val = val;
                    best_cls = c;
                }
            }
            mask_2d.at<uchar>(y, x) = static_cast<uchar>(best_cls);
        }
    }

    // 3. Safe ROI Extraction (Clamps values to strictly prevent OpenCV Assertions)
    int safe_x = std::max(0, std::min(pad_left, mask_2d.cols - 1));
    int safe_y = std::max(0, std::min(pad_top, mask_2d.rows - 1));
    int safe_w = std::max(0, std::min(new_w, mask_2d.cols - safe_x));
    int safe_h = std::max(0, std::min(new_h, mask_2d.rows - safe_y));

    cv::Rect roi(safe_x, safe_y, safe_w, safe_h);
    cv::Mat valid_mask = mask_2d(roi);

    // 4. Resize back to original image dimensions
    cv::Mat final_mask;
    cv::resize(valid_mask, final_mask, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

    return final_mask;
}

cv::Mat draw_segmentation(const cv::Mat &image, const cv::Mat &mask)
{
    cv::Mat color_mask = cv::Mat::zeros(image.size(), CV_8UC3);

    for (int y = 0; y < mask.rows; ++y)
    {
        for (int x = 0; x < mask.cols; ++x)
        {
            int cls_id = mask.at<uchar>(y, x);
            if (cls_id > 0 && cls_id < VOC_COLORS.size())
            {
                color_mask.at<cv::Vec3b>(y, x)[0] = VOC_COLORS[cls_id][0];
                color_mask.at<cv::Vec3b>(y, x)[1] = VOC_COLORS[cls_id][1];
                color_mask.at<cv::Vec3b>(y, x)[2] = VOC_COLORS[cls_id][2];
            }
        }
    }

    cv::Mat blended;
    cv::addWeighted(image, 0.6, color_mask, 0.4, 0.0, blended);
    return blended;
}
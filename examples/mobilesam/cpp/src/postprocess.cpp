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
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

const float SAM_MEAN[3] = {123.675f, 116.28f, 103.53f};
const float SAM_STD[3] = {58.395f, 57.12f, 57.375f};

static std::vector<float> parse_values(std::string values)
{
    std::replace(values.begin(), values.end(), ',', ' ');

    std::stringstream stream(values);
    std::vector<float> parsed;
    float value;

    while (stream >> value)
        parsed.push_back(value);

    return parsed;
}

static bool parse_points(const std::string &values, std::vector<PromptPoint> &points)
{
    std::stringstream stream(values);
    std::string item;

    while (std::getline(stream, item, ';'))
    {
        std::vector<float> parsed = parse_values(item);

        if (parsed.size() != 3 || (parsed[2] != 0.0f && parsed[2] != 1.0f))
            return false;

        points.push_back({parsed[0], parsed[1], parsed[2]});
    }

    return !points.empty() && points.size() <= 2;
}

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

int get_tensor_element_count(const amlnn_tensor_attr &attr)
{
    int count = 1;

    for (int i = 0; i < attr.n_dims; ++i)
        count *= attr.dims[i];

    return count;
}

cv::Mat preprocess(const cv::Mat &image, std::tuple<int, int> new_shape, ImageMeta &meta)
{
    int input_height = std::get<0>(new_shape);
    int input_width = std::get<1>(new_shape);
    float scale = std::min(static_cast<float>(input_height) / image.rows,
                           static_cast<float>(input_width) / image.cols);

    int resized_height = static_cast<int>(image.rows * scale + 0.5f);
    int resized_width = static_cast<int>(image.cols * scale + 0.5f);

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32FC3);

    for (int y = 0; y < resized.rows; ++y)
    {
        cv::Vec3f *row = resized.ptr<cv::Vec3f>(y);

        for (int x = 0; x < resized.cols; ++x)
        {
            for (int c = 0; c < 3; ++c)
                row[x][c] = (row[x][c] - SAM_MEAN[c]) / SAM_STD[c];
        }
    }

    cv::Mat padded = cv::Mat::zeros(input_height, input_width, CV_32FC3);
    resized.copyTo(padded(cv::Rect(0, 0, resized_width, resized_height)));

    meta.original_height = image.rows;
    meta.original_width = image.cols;
    meta.resized_height = resized_height;
    meta.resized_width = resized_width;
    meta.input_height = input_height;
    meta.input_width = input_width;
    meta.scale_x = static_cast<float>(resized_width) / image.cols;
    meta.scale_y = static_cast<float>(resized_height) / image.rows;

    return padded;
}

std::vector<uint8_t> prepare_tensor(const float *data, int total_elements,
                                    const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;

    if (attr.type == AMLNN_TENSOR_FLOAT32)
    {
        tensor_data.resize(total_elements * sizeof(float));
        std::memcpy(tensor_data.data(), data, tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(total_elements * sizeof(int16_t));
        int16_t *dst_ptr = reinterpret_cast<int16_t *>(tensor_data.data());

        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(data[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());

        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(data[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());

        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(data[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, value)));
        }
    }
    else
    {
        std::cerr << "prepare_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img,
                                          const amlnn_tensor_attr &attr)
{
    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        std::cerr << "prepare_input_tensor: Invalid input image" << std::endl;
        return {};
    }

    int total_elements = static_cast<int>(float_img.total() * float_img.channels());
    return prepare_tensor(float_img.ptr<float>(), total_elements, attr);
}

bool build_prompt(const std::string &type, const std::string &values, const ImageMeta &meta,
                  std::vector<float> &point_coords, std::vector<float> &point_labels,
                  Prompt &prompt)
{
    point_coords.assign(4, 0.0f);
    point_labels.assign(2, -1.0f);

    if (type == "point")
    {
        if (!parse_points(values, prompt.points))
            return false;

        for (size_t i = 0; i < prompt.points.size(); ++i)
        {
            point_coords[i * 2] = prompt.points[i].x * meta.scale_x;
            point_coords[i * 2 + 1] = prompt.points[i].y * meta.scale_y;
            point_labels[i] = prompt.points[i].label;
        }

        return true;
    }

    if (type == "box")
    {
        std::vector<float> parsed = parse_values(values);

        if (parsed.size() != 4)
            return false;

        float x1 = std::min(parsed[0], parsed[2]);
        float y1 = std::min(parsed[1], parsed[3]);
        float x2 = std::max(parsed[0], parsed[2]);
        float y2 = std::max(parsed[1], parsed[3]);

        point_coords[0] = x1 * meta.scale_x;
        point_coords[1] = y1 * meta.scale_y;
        point_coords[2] = x2 * meta.scale_x;
        point_coords[3] = y2 * meta.scale_y;
        point_labels[0] = 2.0f;
        point_labels[1] = 3.0f;

        prompt.has_box = true;
        prompt.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        return true;
    }

    return false;
}

MaskResult postprocess(float *mask_data, const amlnn_tensor_attr &mask_attr,
                       float *score_data, int score_elements, const ImageMeta &meta)
{
    MaskResult result;

    if (mask_attr.n_dims != 4 || score_elements <= 0)
    {
        std::cerr << "Unexpected decoder output shape." << std::endl;
        return result;
    }

    int mask_height = mask_attr.dims[1];
    int mask_width = mask_attr.dims[2];
    int num_masks = mask_attr.dims[3];

    result.index = 0;
    result.score = score_data[0];

    for (int i = 1; i < std::min(num_masks, score_elements); ++i)
    {
        if (score_data[i] > result.score)
        {
            result.score = score_data[i];
            result.index = i;
        }
    }

    cv::Mat low_res(mask_height, mask_width, CV_32F);

    for (int y = 0; y < mask_height; ++y)
    {
        float *row = low_res.ptr<float>(y);

        for (int x = 0; x < mask_width; ++x)
            row[x] = mask_data[(y * mask_width + x) * num_masks + result.index];
    }

    cv::Mat input_mask;
    cv::resize(low_res, input_mask, cv::Size(meta.input_width, meta.input_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat cropped = input_mask(cv::Rect(0, 0, meta.resized_width, meta.resized_height));
    cv::Mat original_mask;
    cv::resize(cropped, original_mask, cv::Size(meta.original_width, meta.original_height), 0, 0, cv::INTER_LINEAR);

    result.mask = original_mask > 0.0f;
    return result;
}

cv::Mat draw_result(const cv::Mat &image, const cv::Mat &mask, const Prompt &prompt)
{
    cv::Mat result = image.clone();
    cv::Mat overlay = result.clone();

    overlay.setTo(cv::Scalar(30, 255, 30), mask);
    cv::addWeighted(overlay, 0.55, result, 0.45, 0, result);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(result, contours, -1, cv::Scalar(255, 255, 255), 2);

    if (prompt.has_box)
    {
        cv::Point point1(static_cast<int>(std::round(prompt.box.x)),
                         static_cast<int>(std::round(prompt.box.y)));

        cv::Point point2(static_cast<int>(std::round(prompt.box.x + prompt.box.width)),
                         static_cast<int>(std::round(prompt.box.y + prompt.box.height)));

        cv::rectangle(result, point1, point2, cv::Scalar(255, 166, 88), 2);
    }

    for (const auto &point : prompt.points)
    {
        cv::Point center(static_cast<int>(std::round(point.x)),
                         static_cast<int>(std::round(point.y)));

        cv::Scalar color = point.label == 1.0f ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);

        cv::circle(result, center, 9, cv::Scalar(255, 255, 255), -1);
        cv::circle(result, center, 6, color, -1);
        cv::circle(result, center, 9, cv::Scalar(0, 0, 0), 2);
    }

    return result;
}
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
#include "anchors.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

namespace
{
    float sigmoid(float value)
    {
        value = std::max(-100.0f, std::min(value, 100.0f));
        return 1.0f / (1.0f + std::exp(-value));
    }

    float iou(const Detection &a, const Detection &b)
    {
        float x1 = std::max(a.coords[1], b.coords[1]);
        float y1 = std::max(a.coords[0], b.coords[0]);
        float x2 = std::min(a.coords[3], b.coords[3]);
        float y2 = std::min(a.coords[2], b.coords[2]);
        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_a = std::max(0.0f, a.coords[3] - a.coords[1]) * std::max(0.0f, a.coords[2] - a.coords[0]);
        float area_b = std::max(0.0f, b.coords[3] - b.coords[1]) * std::max(0.0f, b.coords[2] - b.coords[0]);
        return intersection / std::max(area_a + area_b - intersection, 1e-6f);
    }

    std::vector<Detection> weighted_nms(std::vector<Detection> detections, float iou_threshold)
    {
        std::sort(detections.begin(), detections.end(),
                  [](const Detection &a, const Detection &b)
                  { return a.score > b.score; });
        std::vector<Detection> results;

        while (!detections.empty())
        {
            Detection reference = detections.front();
            std::vector<Detection> group = {reference};
            std::vector<Detection> remaining;

            for (size_t i = 1; i < detections.size(); ++i)
            {
                const Detection &detection = detections[i];
                if (iou(reference, detection) > iou_threshold)
                    group.push_back(detection);
                else
                    remaining.push_back(detection);
            }

            Detection merged;
            float weight_sum = 0.0f;
            for (const auto &detection : group)
            {
                weight_sum += detection.score;
                merged.score = std::max(merged.score, detection.score);
                for (int i = 0; i < NUM_COORDS; ++i)
                    merged.coords[i] += detection.coords[i] * detection.score;
            }
            for (float &value : merged.coords)
                value /= std::max(weight_sum, 1e-6f);

            results.push_back(merged);
            detections.swap(remaining);
        }
        return results;
    }

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

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape)
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
    float scale = std::min(static_cast<float>(target_h) / orig_h, static_cast<float>(target_w) / orig_w);
    int new_h = static_cast<int>(std::round(orig_h * scale));
    int new_w = static_cast<int>(std::round(orig_w * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_h = target_h - new_h;
    int pad_w = target_w - new_w;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;

    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32FC3, 1.0 / 127.5, -1.0);
    return std::make_tuple(img_float, scale, std::make_tuple(pad_left, pad_top));
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

std::vector<Detection> postprocess(
    const std::vector<float *> &out_ptrs,
    const std::vector<std::vector<int>> &out_shapes,
    int input_h, int input_w,
    int original_h, int original_w,
    std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
    float conf_threshold, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    // Calculate the total number of elements from each output shape.
    int box_elements = 1;
    for (int dim : out_shapes[0])
        box_elements *= dim;

    int score_elements = 1;
    for (int dim : out_shapes[1])
        score_elements *= dim;

    // Output 0: [1, 1, 2254, 12]
    // Output 1: [1, 1, 2254, 1]
    if (box_elements != NUM_ANCHORS * NUM_COORDS)
    {
        std::cerr << "Unexpected box output shape. Expected "
                  << NUM_ANCHORS * NUM_COORDS
                  << " elements, got " << box_elements << std::endl;
        return {};
    }

    if (score_elements != NUM_ANCHORS)
    {
        std::cerr << "Unexpected score output shape. Expected "
                  << NUM_ANCHORS
                  << " elements, got " << score_elements << std::endl;
        return {};
    }

    float *raw_boxes = out_ptrs[0];
    float *raw_scores = out_ptrs[1];

    std::vector<Detection> detections;

    for (int i = 0; i < NUM_ANCHORS; ++i)
    {
        float raw_score = std::max(-100.0f, std::min(raw_scores[i], 100.0f));
        float score = 1.0f / (1.0f + std::exp(-raw_score));

        if (score <= conf_threshold)
            continue;

        int anchor_idx = i * 4;

        float anc_x = anchors[anchor_idx];
        float anc_y = anchors[anchor_idx + 1];
        float anc_w = anchors[anchor_idx + 2];
        float anc_h = anchors[anchor_idx + 3];

        const float *box = raw_boxes + i * NUM_COORDS;

        // Decode the bounding box using the normalized anchor.
        float cx = box[0] / input_w * anc_w + anc_x;
        float cy = box[1] / input_h * anc_h + anc_y;
        float width = box[2] / input_w * anc_w;
        float height = box[3] / input_h * anc_h;

        Detection detection;

        // Detection coordinate format:
        // [ymin, xmin, ymax, xmax, kp0_x, kp0_y, ..., kp3_x, kp3_y]
        detection.coords[0] = cy - height / 2.0f;
        detection.coords[1] = cx - width / 2.0f;
        detection.coords[2] = cy + height / 2.0f;
        detection.coords[3] = cx + width / 2.0f;

        // Decode the four detector keypoints.
        for (int k = 0; k < 4; ++k)
        {
            detection.coords[4 + k * 2] =
                box[4 + k * 2] / input_w * anc_w + anc_x;

            detection.coords[5 + k * 2] =
                box[5 + k * 2] / input_h * anc_h + anc_y;
        }

        detection.score = score;
        detections.push_back(detection);
    }

    detections = weighted_nms(std::move(detections), iou_threshold);

    // Remove letterbox and normalize coordinates relative to the original image.
    for (auto &detection : detections)
    {
        // xmin, xmax and keypoint x coordinates
        for (int index : {1, 3, 4, 6, 8, 10})
        {
            detection.coords[index] =
                (detection.coords[index] * input_w - pad_left) / scale / original_w;
        }

        // ymin, ymax and keypoint y coordinates
        for (int index : {0, 2, 5, 7, 9, 11})
        {
            detection.coords[index] =
                (detection.coords[index] * input_h - pad_top) / scale / original_h;
        }

        for (int i = 0; i < 4; ++i)
        {
            detection.coords[i] = std::max(
                0.0f, std::min(detection.coords[i], 1.0f));
        }
    }

    return detections;
}

bool save_detections(const std::string &path, const std::vector<Detection> &detections)
{
    std::ofstream file(path);
    if (!file.is_open())
        return false;
    file << std::fixed << std::setprecision(8);
    for (const auto &detection : detections)
    {
        for (float value : detection.coords)
            file << value << ' ';
        file << detection.score << '\n';
    }
    return true;
}

cv::Mat draw_detections(const cv::Mat &image, const std::vector<Detection> &detections)
{
    cv::Mat drawn = image.clone();
    for (const auto &detection : detections)
    {
        int x1 = static_cast<int>(detection.coords[1] * image.cols);
        int y1 = static_cast<int>(detection.coords[0] * image.rows);
        int x2 = static_cast<int>(detection.coords[3] * image.cols);
        int y2 = static_cast<int>(detection.coords[2] * image.rows);
        cv::rectangle(drawn, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);
        cv::putText(drawn, "pose: " + cv::format("%.2f", detection.score), {x1, std::max(15, y1 - 5)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
    }
    return drawn;
}

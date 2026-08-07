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
#include <string>
#include <utility>

const std::vector<std::string> KEYPOINT_NAMES = {
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr",
    "l_hip", "r_hip", "l_kn", "r_kn", "l_an", "r_an"};

const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};

const int STRIDES[3] = {8, 16, 32};

static float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

static cv::Scalar get_color(float confidence)
{
    float hue = std::fmod(confidence * 137.508f, 360.0f);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

static float compute_iou(const Detection &det1, const Detection &det2)
{
    float x1 = std::max(det1.x1, det2.x1);
    float y1 = std::max(det1.y1, det2.y1);
    float x2 = std::min(det1.x2, det2.x2);
    float y2 = std::min(det1.y2, det2.y2);
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = std::max(0.0f, det1.x2 - det1.x1) * std::max(0.0f, det1.y2 - det1.y1);
    float area2 = std::max(0.0f, det2.x2 - det2.x1) * std::max(0.0f, det2.y2 - det2.y1);
    float union_area = area1 + area2 - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static std::vector<Detection> nms(
    const std::vector<Detection> &detections, float iou_threshold)
{
    if (detections.empty())
        return {};

    std::vector<Detection> sorted_detections = detections;
    std::sort(
        sorted_detections.begin(), sorted_detections.end(),
        [](const Detection &a, const Detection &b)
        { return a.score > b.score; });

    std::vector<Detection> final_detections;
    std::vector<bool> removed(sorted_detections.size(), false);
    for (size_t i = 0; i < sorted_detections.size(); ++i)
    {
        if (removed[i])
            continue;

        final_detections.push_back(sorted_detections[i]);
        for (size_t j = i + 1; j < sorted_detections.size(); ++j)
        {
            if (!removed[j] && compute_iou(sorted_detections[i], sorted_detections[j]) > iou_threshold)
                removed[j] = true;
        }
    }

    return final_detections;
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

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(
    cv::Mat img, std::tuple<int, int> new_shape)
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

    int original_h = img.rows;
    int original_w = img.cols;
    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);
    float scale = std::min(
        static_cast<float>(target_h) / original_h,
        static_cast<float>(target_w) / original_w);
    int resized_h = static_cast<int>(std::round(original_h * scale));
    int resized_w = static_cast<int>(std::round(original_w * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    int pad_h = target_h - resized_h;
    int pad_w = target_w - resized_w;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;

    cv::Mat img_padded;
    cv::copyMakeBorder(
        img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32F, 1.0 / 255.0);
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

std::vector<Detection> postprocess(
    const std::vector<float *> &out_ptrs, const std::vector<std::vector<int>> &out_shapes,
    int input_h, int input_w, std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
    float conf_thresh, float iou_threshold, int reg_max)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));
    std::vector<Detection> detections_orig;

    float safe_thresh = std::max(1e-5f, std::min(conf_thresh, 1.0f - 1e-5f));
    float inv_thresh = std::log(safe_thresh / (1.0f - safe_thresh));
    const int detection_channels = reg_max * 4 + 1;
    const int keypoint_channels = NUM_KEYPOINTS * 3;

    for (int s = 0; s < 3; ++s)
    {
        int detection_idx = s * 2;
        int keypoint_idx = s * 2 + 1;
        int stride = STRIDES[s];

        float *detection_data = out_ptrs[detection_idx];
        float *keypoint_data = out_ptrs[keypoint_idx];
        const auto &detection_shape = out_shapes[detection_idx];
        const auto &keypoint_shape = out_shapes[keypoint_idx];

        int height = 1;
        int width = 1;
        int channels = 1;

        if (detection_shape.size() == 4)
        {
            height = detection_shape[1];
            width = detection_shape[2];
            channels = detection_shape[3];
        }
        else if (detection_shape.size() == 3)
        {
            height = detection_shape[0];
            width = detection_shape[1];
            channels = detection_shape[2];
        }
        else
        {
            std::cerr << "Unexpected detection output shape for output " << detection_idx << std::endl;
            continue;
        }

        if (channels != detection_channels)
        {
            std::cerr << "Detection output " << detection_idx << " expected " << detection_channels << " channels, got " << channels << std::endl;
            continue;
        }

        int keypoint_height = 1;
        int keypoint_width = 1;
        int keypoint_output_channels = 1;

        if (keypoint_shape.size() == 4)
        {
            keypoint_height = keypoint_shape[1];
            keypoint_width = keypoint_shape[2];
            keypoint_output_channels = keypoint_shape[3];
        }
        else if (keypoint_shape.size() == 3)
        {
            keypoint_height = keypoint_shape[0];
            keypoint_width = keypoint_shape[1];
            keypoint_output_channels = keypoint_shape[2];
        }
        else
        {
            std::cerr << "Unexpected keypoint output shape for output " << keypoint_idx << std::endl;
            continue;
        }

        if (keypoint_height != height || keypoint_width != width || keypoint_output_channels != keypoint_channels)
        {
            std::cerr << "Keypoint output " << keypoint_idx << " does not match [" << height << ", " << width << ", " << keypoint_channels << "]" << std::endl;
            continue;
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int cell_idx = y * width + x;
                const float *detection_cell = detection_data + cell_idx * detection_channels;
                float raw_confidence = detection_cell[reg_max * 4];

                if (raw_confidence <= inv_thresh)
                    continue;

                float bbox_distances[4] = {};

                for (int side = 0; side < 4; ++side)
                {
                    const float *dfl_data = detection_cell + side * reg_max;
                    float max_value = dfl_data[0];

                    for (int i = 1; i < reg_max; ++i)
                        max_value = std::max(max_value, dfl_data[i]);

                    float sum = 0.0f;
                    float weighted_sum = 0.0f;

                    for (int i = 0; i < reg_max; ++i)
                    {
                        float value = std::exp(dfl_data[i] - max_value);
                        sum += value;
                        weighted_sum += value * static_cast<float>(i);
                    }

                    bbox_distances[side] = weighted_sum / sum;
                }

                float center_x = (static_cast<float>(x) + 0.5f) * stride;
                float center_y = (static_cast<float>(y) + 0.5f) * stride;

                Detection detection;
                detection.x1 = (center_x - bbox_distances[0] * stride - pad_left) / scale;
                detection.y1 = (center_y - bbox_distances[1] * stride - pad_top) / scale;
                detection.x2 = (center_x + bbox_distances[2] * stride - pad_left) / scale;
                detection.y2 = (center_y + bbox_distances[3] * stride - pad_top) / scale;
                detection.x1 = std::max(0.0f, std::min(detection.x1, static_cast<float>(std::get<0>(input_tuple).cols - 1)));
                detection.y1 = std::max(0.0f, std::min(detection.y1, static_cast<float>(std::get<0>(input_tuple).rows - 1)));
                detection.x2 = std::max(0.0f, std::min(detection.x2, static_cast<float>(std::get<0>(input_tuple).cols - 1)));
                detection.y2 = std::max(0.0f, std::min(detection.y2, static_cast<float>(std::get<0>(input_tuple).rows - 1)));
                detection.score = sigmoid(raw_confidence);

                const float *keypoint_cell = keypoint_data + cell_idx * keypoint_channels;

                for (int keypoint_idx = 0; keypoint_idx < NUM_KEYPOINTS; ++keypoint_idx)
                {
                    float raw_x = keypoint_cell[keypoint_idx * 3];
                    float raw_y = keypoint_cell[keypoint_idx * 3 + 1];
                    float raw_keypoint_confidence = keypoint_cell[keypoint_idx * 3 + 2];

                    float keypoint_x = ((raw_x * 2.0f + static_cast<float>(x)) * stride - pad_left) / scale;
                    float keypoint_y = ((raw_y * 2.0f + static_cast<float>(y)) * stride - pad_top) / scale;

                    detection.keypoints[keypoint_idx] = cv::Point2f(keypoint_x, keypoint_y);
                    detection.keypoint_confidences[keypoint_idx] = sigmoid(raw_keypoint_confidence);
                }

                detections_orig.push_back(detection);
            }
        }
    }

    return nms(detections_orig, iou_threshold);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection> &detections)
{
    cv::Mat drawn_image = image.clone();
    int image_height = drawn_image.rows;
    int image_width = drawn_image.cols;

    for (const auto &detection : detections)
    {
        cv::Scalar color = get_color(detection.score);

        cv::rectangle(
            drawn_image,
            cv::Point(static_cast<int>(detection.x1), static_cast<int>(detection.y1)),
            cv::Point(static_cast<int>(detection.x2), static_cast<int>(detection.y2)),
            color, 2);

        std::string label = "conf: " + cv::format("%.2f", detection.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        int label_x = std::max(0, static_cast<int>(detection.x1));
        int label_y = std::max(
            static_cast<int>(detection.y1) - 5, text_size.height + baseline);
        cv::rectangle(
            drawn_image, cv::Point(label_x, label_y - text_size.height - baseline),
            cv::Point(label_x + text_size.width, label_y + baseline), color, cv::FILLED);

        int brightness = static_cast<int>((color[0] + color[1] + color[2]) / 3);
        cv::Scalar text_color = brightness < 128
                                    ? cv::Scalar(255, 255, 255)
                                    : cv::Scalar(0, 0, 0);
        cv::putText(
            drawn_image, label, cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX,
            0.6, text_color, 1, cv::LINE_AA);

        // Draw the skeleton before keypoints so the points remain visible.
        for (const auto &connection : SKELETON)
        {
            int start_idx = connection.first;
            int end_idx = connection.second;
            if (detection.keypoint_confidences[start_idx] <= KEYPOINT_THRESHOLD ||
                detection.keypoint_confidences[end_idx] <= KEYPOINT_THRESHOLD)
            {
                continue;
            }

            cv::Point2f start = detection.keypoints[start_idx];
            cv::Point2f end = detection.keypoints[end_idx];
            if (start.x < 0 || start.x >= image_width || start.y < 0 || start.y >= image_height ||
                end.x < 0 || end.x >= image_width || end.y < 0 || end.y >= image_height)
            {
                continue;
            }

            cv::Point start_point(static_cast<int>(start.x), static_cast<int>(start.y));
            cv::Point end_point(static_cast<int>(end.x), static_cast<int>(end.y));
            cv::line(drawn_image, start_point, end_point, cv::Scalar(0, 255, 0), 2);
        }

        for (int keypoint_idx = 0; keypoint_idx < NUM_KEYPOINTS; ++keypoint_idx)
        {
            if (detection.keypoint_confidences[keypoint_idx] <= KEYPOINT_THRESHOLD)
                continue;

            cv::Point2f keypoint = detection.keypoints[keypoint_idx];
            if (keypoint.x < 0 || keypoint.x >= image_width ||
                keypoint.y < 0 || keypoint.y >= image_height)
            {
                continue;
            }

            cv::Point point(static_cast<int>(keypoint.x), static_cast<int>(keypoint.y));
            cv::circle(drawn_image, point, 4, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::putText(
                drawn_image, KEYPOINT_NAMES[keypoint_idx],
                cv::Point(point.x + 5, point.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    return drawn_image;
}
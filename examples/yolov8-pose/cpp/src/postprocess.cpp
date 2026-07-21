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

const float KEYPOINT_THRESHOLD = 0.5f;

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
    std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
    float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));
    std::vector<Detection> detections_orig;

    if (out_ptrs.size() != 4 || out_shapes.size() != 4)
    {
        std::cerr << "Expected exactly 4 YOLOv8-Pose outputs." << std::endl;
        return {};
    }

    // Shapes after singleton removal: [4,N], [N], [N,17], and [2,N,17].
    if (out_shapes[1].size() != 1)
    {
        std::cerr << "Unexpected detection confidence output shape." << std::endl;
        return {};
    }

    int num_predictions = out_shapes[1][0];
    if (out_shapes[0].size() != 2 || out_shapes[0][0] != 4 ||
        out_shapes[0][1] != num_predictions)
    {
        std::cerr << "Unexpected bbox output shape." << std::endl;
        return {};
    }

    if (out_shapes[2].size() != 2 || out_shapes[2][0] != num_predictions ||
        out_shapes[2][1] != NUM_KEYPOINTS)
    {
        std::cerr << "Unexpected keypoint confidence output shape." << std::endl;
        return {};
    }

    if (out_shapes[3].size() != 3 || out_shapes[3][0] != 2 ||
        out_shapes[3][1] != num_predictions || out_shapes[3][2] != NUM_KEYPOINTS)
    {
        std::cerr << "Unexpected keypoint coordinate output shape." << std::endl;
        return {};
    }

    float *bbox_data = out_ptrs[0];
    float *confidence_data = out_ptrs[1];
    float *keypoint_confidence_data = out_ptrs[2];
    float *keypoint_data = out_ptrs[3];
    int keypoint_plane_size = num_predictions * NUM_KEYPOINTS;

    // Outputs are decoded XYWH boxes, probabilities, and keypoint coordinates.
    for (int prediction_idx = 0; prediction_idx < num_predictions; ++prediction_idx)
    {
        float confidence = confidence_data[prediction_idx];
        if (confidence <= conf_thresh)
            continue;

        float center_x = bbox_data[prediction_idx];
        float center_y = bbox_data[num_predictions + prediction_idx];
        float width = bbox_data[num_predictions * 2 + prediction_idx];
        float height = bbox_data[num_predictions * 3 + prediction_idx];

        Detection detection;
        detection.x1 = (center_x - width * 0.5f - pad_left) / scale;
        detection.y1 = (center_y - height * 0.5f - pad_top) / scale;
        detection.x2 = (center_x + width * 0.5f - pad_left) / scale;
        detection.y2 = (center_y + height * 0.5f - pad_top) / scale;
        detection.score = confidence;

        int keypoint_offset = prediction_idx * NUM_KEYPOINTS;
        for (int keypoint_idx = 0; keypoint_idx < NUM_KEYPOINTS; ++keypoint_idx)
        {
            int index = keypoint_offset + keypoint_idx;
            float x = (keypoint_data[index] - pad_left) / scale;
            float y = (keypoint_data[keypoint_plane_size + index] - pad_top) / scale;
            detection.keypoints[keypoint_idx] = cv::Point2f(x, y);
            detection.keypoint_confidences[keypoint_idx] = keypoint_confidence_data[index];
        }

        detections_orig.push_back(detection);
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
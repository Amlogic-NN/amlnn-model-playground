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
#include <unordered_map>

const std::vector<std::string> DOTA_CLASSES = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"};

const int STRIDES[3] = {8, 16, 32};

static float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

static cv::Scalar get_color(int class_id)
{
    float hue = std::fmod(class_id * 137.508f, 360.0f);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

static cv::Rect2f get_enclosing_rect(const Detection &detection)
{
    float min_x = detection.corners[0].x;
    float min_y = detection.corners[0].y;
    float max_x = detection.corners[0].x;
    float max_y = detection.corners[0].y;

    for (int i = 1; i < 4; ++i)
    {
        min_x = std::min(min_x, detection.corners[i].x);
        min_y = std::min(min_y, detection.corners[i].y);
        max_x = std::max(max_x, detection.corners[i].x);
        max_y = std::max(max_y, detection.corners[i].y);
    }

    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}

static float compute_rotated_iou(const Detection &det1, const Detection &det2)
{
    std::vector<cv::Point2f> points1(det1.corners.begin(), det1.corners.end());
    std::vector<cv::Point2f> points2(det2.corners.begin(), det2.corners.end());
    cv::RotatedRect rect1 = cv::minAreaRect(points1);
    cv::RotatedRect rect2 = cv::minAreaRect(points2);

    std::vector<cv::Point2f> intersection_points;
    int intersection_type = cv::rotatedRectangleIntersection(rect1, rect2, intersection_points);

    if (intersection_type == cv::INTERSECT_NONE || intersection_points.size() < 3)
        return 0.0f;

    std::vector<cv::Point2f> intersection_hull;
    cv::convexHull(intersection_points, intersection_hull);
    float intersection_area = static_cast<float>(std::abs(cv::contourArea(intersection_hull)));
    float area1 = rect1.size.width * rect1.size.height;
    float area2 = rect2.size.width * rect2.size.height;
    float union_area = area1 + area2 - intersection_area;

    return union_area > 0.0f ? intersection_area / union_area : 0.0f;
}

static std::vector<Detection> nms_by_class(
    const std::vector<Detection> &detections, float iou_threshold)
{
    if (detections.empty())
        return {};

    std::vector<Detection> final_detections;
    std::unordered_map<int, std::vector<Detection>> class_detections;
    for (const auto &detection : detections)
        class_detections[detection.class_id].push_back(detection);

    for (auto &[class_id, class_dets] : class_detections)
    {
        std::sort(
            class_dets.begin(), class_dets.end(),
            [](const Detection &a, const Detection &b)
            { return a.score > b.score; });
        std::vector<bool> removed(class_dets.size(), false);

        for (size_t i = 0; i < class_dets.size(); ++i)
        {
            if (removed[i])
                continue;

            final_detections.push_back(class_dets[i]);
            for (size_t j = i + 1; j < class_dets.size(); ++j)
            {
                if (!removed[j] &&
                    compute_rotated_iou(class_dets[i], class_dets[j]) > iou_threshold)
                {
                    removed[j] = true;
                }
            }
        }
    }

    std::sort(
        final_detections.begin(), final_detections.end(),
        [](const Detection &a, const Detection &b)
        { return a.score > b.score; });
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
    float conf_thresh, float iou_threshold, int reg_max)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));
    std::vector<Detection> detections_orig;

    float safe_thresh = std::max(1e-5f, std::min(conf_thresh, 1.0f - 1e-5f));
    float inv_thresh = std::log(safe_thresh / (1.0f - safe_thresh));
    int num_classes = static_cast<int>(DOTA_CLASSES.size());
    int dfl_channels = 4 * reg_max;

    // Output order: [DFL_8, ANGLE_8, CLS_8, DFL_16, ANGLE_16, CLS_16, DFL_32, ANGLE_32, CLS_32].
    for (int s = 0; s < 3; ++s)
    {
        int dfl_idx = s * 3;
        int angle_idx = s * 3 + 1;
        int class_idx = s * 3 + 2;
        int stride = STRIDES[s];

        float *dfl_data = out_ptrs[dfl_idx];
        float *angle_data = out_ptrs[angle_idx];
        float *class_data = out_ptrs[class_idx];

        const auto &dfl_shape = out_shapes[dfl_idx];
        const auto &angle_shape = out_shapes[angle_idx];
        const auto &class_shape = out_shapes[class_idx];

        int height = 1;
        int width = 1;
        int channels = 1;

        if (dfl_shape.size() == 4)
        {
            height = dfl_shape[1];
            width = dfl_shape[2];
            channels = dfl_shape[3];
        }
        else if (dfl_shape.size() == 3)
        {
            height = dfl_shape[0];
            width = dfl_shape[1];
            channels = dfl_shape[2];
        }
        else
        {
            std::cerr << "Unexpected DFL output shape for output " << dfl_idx << std::endl;
            continue;
        }

        if (channels != dfl_channels)
        {
            std::cerr << "DFL output " << dfl_idx << " expected " << dfl_channels << " channels, got " << channels << std::endl;
            continue;
        }

        int angle_height = 1;
        int angle_width = 1;
        int angle_channels = 1;

        if (angle_shape.size() == 4)
        {
            angle_height = angle_shape[1];
            angle_width = angle_shape[2];
            angle_channels = angle_shape[3];
        }
        else if (angle_shape.size() == 3)
        {
            angle_height = angle_shape[0];
            angle_width = angle_shape[1];
            angle_channels = angle_shape[2];
        }
        else if (angle_shape.size() == 2)
        {
            angle_height = angle_shape[0];
            angle_width = angle_shape[1];
            angle_channels = 1;
        }
        else
        {
            std::cerr << "Unexpected angle output shape for output " << angle_idx << std::endl;
            continue;
        }

        int class_height = 1;
        int class_width = 1;
        int class_channels = 1;

        if (class_shape.size() == 4)
        {
            class_height = class_shape[1];
            class_width = class_shape[2];
            class_channels = class_shape[3];
        }
        else if (class_shape.size() == 3)
        {
            class_height = class_shape[0];
            class_width = class_shape[1];
            class_channels = class_shape[2];
        }
        else
        {
            std::cerr << "Unexpected class output shape for output " << class_idx << std::endl;
            continue;
        }

        if (angle_height != height || angle_width != width || angle_channels != 1)
        {
            std::cerr << "Angle output " << angle_idx << " does not match [" << height << ", " << width << ", 1]" << std::endl;
            continue;
        }

        if (class_height != height || class_width != width || class_channels != num_classes)
        {
            std::cerr << "Class output " << class_idx << " does not match [" << height << ", " << width << ", " << num_classes << "]" << std::endl;
            continue;
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int cell_idx = y * width + x;
                const float *class_cell = class_data + cell_idx * num_classes;
                float max_raw_score = -1e9f;
                int class_id = -1;

                for (int c = 0; c < num_classes; ++c)
                {
                    if (class_cell[c] > max_raw_score)
                    {
                        max_raw_score = class_cell[c];
                        class_id = c;
                    }
                }

                // Skip DFL and angle decoding for low-confidence cells.
                if (max_raw_score <= inv_thresh)
                    continue;

                const float *dfl_cell = dfl_data + cell_idx * dfl_channels;
                float distances[4] = {};

                // Decode DFL in left, top, right, bottom order.
                for (int side = 0; side < 4; ++side)
                {
                    const float *distribution = dfl_cell + side * reg_max;
                    float max_value = distribution[0];

                    for (int i = 1; i < reg_max; ++i)
                        max_value = std::max(max_value, distribution[i]);

                    float sum = 0.0f;
                    float weighted_sum = 0.0f;

                    for (int i = 0; i < reg_max; ++i)
                    {
                        float probability = std::exp(distribution[i] - max_value);
                        sum += probability;
                        weighted_sum += probability * static_cast<float>(i);
                    }

                    distances[side] = weighted_sum / sum;
                }

                float left = distances[0];
                float top = distances[1];
                float right = distances[2];
                float bottom = distances[3];

                // YOLOv8-OBB angle range: [-pi/4, 3pi/4].
                float angle_radians = (sigmoid(angle_data[cell_idx]) - 0.25f) * static_cast<float>(CV_PI);
                float cos_angle = std::cos(angle_radians);
                float sin_angle = std::sin(angle_radians);

                float offset_x = (right - left) * 0.5f;
                float offset_y = (bottom - top) * 0.5f;
                float center_x = (static_cast<float>(x) + 0.5f + offset_x * cos_angle - offset_y * sin_angle) * stride;
                float center_y = (static_cast<float>(y) + 0.5f + offset_x * sin_angle + offset_y * cos_angle) * stride;
                float width = (left + right) * stride;
                float height_box = (top + bottom) * stride;

                center_x = (center_x - pad_left) / scale;
                center_y = (center_y - pad_top) / scale;
                width /= scale;
                height_box /= scale;

                if (width <= 0.0f || height_box <= 0.0f)
                    continue;

                float angle_degrees = angle_radians * 180.0f / static_cast<float>(CV_PI);
                cv::RotatedRect rotated_rect(cv::Point2f(center_x, center_y), cv::Size2f(width, height_box), angle_degrees);

                Detection detection;
                rotated_rect.points(detection.corners.data());
                detection.score = sigmoid(max_raw_score);
                detection.class_id = class_id;
                detections_orig.push_back(detection);
            }
        }
    }

    return nms_by_class(detections_orig, iou_threshold);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection> &detections)
{
    cv::Mat drawn_image = image.clone();

    for (const auto &detection : detections)
    {
        int class_id = detection.class_id;
        cv::Scalar color = get_color(class_id);
        std::vector<cv::Point> int_corners(4);
        int top_idx = 0;
        float min_y = detection.corners[0].y;

        for (int i = 0; i < 4; ++i)
        {
            int_corners[i] = cv::Point(
                static_cast<int>(std::round(detection.corners[i].x)),
                static_cast<int>(std::round(detection.corners[i].y)));

            if (detection.corners[i].y < min_y)
            {
                min_y = detection.corners[i].y;
                top_idx = i;
            }
        }

        std::vector<std::vector<cv::Point>> polygons = {int_corners};
        cv::polylines(drawn_image, polygons, true, color, 2);

        std::string class_name = class_id >= 0 && class_id < static_cast<int>(DOTA_CLASSES.size())
                                     ? DOTA_CLASSES[class_id]
                                     : "Unknown";
        std::string label = class_name + ": " + cv::format("%.2f", detection.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        int label_x = std::max(0, int_corners[top_idx].x);
        int label_y = std::max(
            int_corners[top_idx].y - 5, text_size.height + baseline);
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
    }

    return drawn_image;
}
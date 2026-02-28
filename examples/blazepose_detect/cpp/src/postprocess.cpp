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
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>

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

// SHOW class names (1 classes)
const char *SHOW_CLASSES[1] = {"pose"};

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

void decode_boxes(const float *ori_boxes, std::vector<std::vector<float>> &boxes)
{
    const float x_scale = 224.0f;
    const float y_scale = 224.0f;
    const float h_scale = 224.0f;
    const float w_scale = 224.0f;
    boxes.resize(NUM_ANCHORS, std::vector<float>(NUM_COORDS, 0.0f));
    for (int i = 0; i < NUM_ANCHORS; ++i)
    {
        float x_center = ori_boxes[i * NUM_COORDS + 0] / x_scale * anchors[i * 4 + 2] + anchors[i * 4 + 0];
        float y_center = ori_boxes[i * NUM_COORDS + 1] / y_scale * anchors[i * 4 + 3] + anchors[i * 4 + 1];
        float w = ori_boxes[i * NUM_COORDS + 2] / w_scale * anchors[i * 4 + 2];
        float h = ori_boxes[i * NUM_COORDS + 3] / h_scale * anchors[i * 4 + 3];
        boxes[i][0] = y_center - h / 2.0f;
        boxes[i][1] = x_center - w / 2.0f;
        boxes[i][2] = y_center + h / 2.0f;
        boxes[i][3] = x_center + w / 2.0f;
        for (int k = 0; k < 4; ++k)
        {
            int offset = 4 + k * 2;
            float keypoint_x = ori_boxes[i * NUM_COORDS + offset] / x_scale * anchors[i * 4 + 2] + anchors[i * 4 + 0];
            float keypoint_y = ori_boxes[i * NUM_COORDS + offset + 1] / y_scale * anchors[i * 4 + 3] + anchors[i * 4 + 1];
            boxes[i][offset] = keypoint_x;
            boxes[i][offset + 1] = keypoint_y;
        }
    }
}

void convert_output_to_detections(const float *ori_boxes, const float *ori_scores, std::vector<BlazePoseDetection> &detections, float min_score_thresh = 0.3f)
{
    std::vector<std::vector<float>> decoded_boxes;
    decode_boxes(ori_boxes, decoded_boxes);
    detections.clear();
    for (int i = 0; i < NUM_ANCHORS; ++i)
    {
        float s = sigmoid(std::min(std::max(ori_scores[i], -100.0f), 100.0f));
        if (s < min_score_thresh)
            continue;
        BlazePoseDetection det;
        for (int j = 0; j < NUM_COORDS; ++j)
            det.coords[j] = decoded_boxes[i][j];
        det.coords[NUM_COORDS] = s;
        detections.push_back(det);
    }
}

static inline float iou(const float *a, const float *b)
{
    float xA = std::max(a[1], b[1]);
    float yA = std::max(a[0], b[0]);
    float xB = std::min(a[3], b[3]);
    float yB = std::min(a[2], b[2]);
    float interW = std::max(0.0f, xB - xA);
    float interH = std::max(0.0f, yB - yA);
    float inter = interW * interH;
    float areaA = (a[3] - a[1]) * (a[2] - a[0]);
    float areaB = (b[3] - b[1]) * (b[2] - b[0]);
    float unionAB = areaA + areaB - inter;
    if (unionAB <= 0.0f)
        return 0.0f;
    return inter / unionAB;
}

void weighted_nms(
    std::vector<BlazePoseDetection> &detections, std::vector<BlazePoseDetection> &output, float iou_threshold = 0.3f)
{
    output.clear();
    if (detections.empty())
        return;
    std::sort(detections.begin(), detections.end(),
              [](const BlazePoseDetection &a, const BlazePoseDetection &b)
              {
                  return a.coords[NUM_COORDS] > b.coords[NUM_COORDS];
              });
    std::vector<bool> removed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (removed[i])
            continue;
        std::vector<size_t> overlap_indices;
        overlap_indices.push_back(i);
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (removed[j])
                continue;
            if (iou(detections[i].coords, detections[j].coords) > iou_threshold)
                overlap_indices.push_back(j);
        }
        float total_score = 0.0f;
        std::vector<float> weighted(NUM_COORDS, 0.0f);
        for (size_t idx : overlap_indices)
        {
            float score = detections[idx].coords[NUM_COORDS];
            total_score += score;
            for (int k = 0; k < NUM_COORDS; ++k)
                weighted[k] += detections[idx].coords[k] * score;
            removed[idx] = true;
        }
        BlazePoseDetection wdet;
        for (int k = 0; k < NUM_COORDS; ++k)
            wdet.coords[k] = weighted[k] / total_score;
        wdet.coords[NUM_COORDS] = total_score / overlap_indices.size();
        output.push_back(wdet);
    }
}

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape)
{
    cv::Mat img_rgb;

    if (img.empty())
    {
        LOGE("Preprocess received empty image");
        return {};
    }

    // Convert to RGB
    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int orig_h = img.rows;
    int orig_w = img.cols;
    float scale = std::min(static_cast<float>(std::get<0>(new_shape)) / orig_h,
                           static_cast<float>(std::get<1>(new_shape)) / orig_w);
    int new_h = static_cast<int>(round(orig_h * scale));
    int new_w = static_cast<int>(round(orig_w * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_h = std::get<0>(new_shape) - new_h;
    int pad_w = std::get<1>(new_shape) - new_w;
    int pad_left = static_cast<int>(round(pad_w / 2.0 - 0.1));
    int pad_right = static_cast<int>(round(pad_w / 2.0 + 0.1));
    int pad_top = static_cast<int>(round(pad_h / 2.0 - 0.1));
    int pad_bottom = static_cast<int>(round(pad_h / 2.0 + 0.1));

    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32F, 1.0 / 127.5, -1.0);

    scale = 1.0f / scale;
    int pad_orig_h = static_cast<int>(pad_top * scale);
    int pad_orig_w = static_cast<int>(pad_left * scale);

    return std::make_tuple(img_float, scale, std::make_tuple(pad_orig_h, pad_orig_w));
}

cv::Mat quantize_input(const cv::Mat &float_img, float scale, int8_t zero_point)
{
    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        LOGE("quantize_input: Invalid input image (must be CV_32FC3)");
        return cv::Mat();
    }

    cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8SC3);
    const float *src_ptr = (const float *)float_img.data;
    int8_t *dst_ptr = (int8_t *)quantized_img.data;

    int total_elements = float_img.total() * float_img.channels();
    for (int i = 0; i < total_elements; ++i)
    {
        dst_ptr[i] = static_cast<int8_t>(std::round(src_ptr[i] / scale + zero_point));
    }

    return quantized_img;
}

void denorm_detections(std::vector<float> &detection, float scale, const float pad[2])
{
    detection[0] = detection[0] * scale * 224.0f - pad[0];
    detection[1] = detection[1] * scale * 224.0f - pad[1];
    detection[2] = detection[2] * scale * 224.0f - pad[0];
    detection[3] = detection[3] * scale * 224.0f - pad[1];

    for (size_t k = 4; k + 1 < detection.size(); k += 2)
    {
        detection[k] = detection[k] * scale * 224.0f - pad[1];
        detection[k + 1] = detection[k + 1] * scale * 224.0f - pad[0];
    }
}

std::vector<BlazePoseDetection> postprocess(float *ori_boxes, float *ori_scores,
                                            std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                            float conf_threshold, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));
    float pad[2] = {static_cast<float>(pad_left), static_cast<float>(pad_top)};

    std::vector<BlazePoseDetection> detections;
    convert_output_to_detections(ori_boxes, ori_scores, detections, conf_threshold);

    std::vector<BlazePoseDetection> filtered;
    weighted_nms(detections, filtered, iou_threshold);

    int pose_num = filtered.size();
    for (size_t b = 0; b < pose_num; ++b)
    {
        std::vector<float> coords(filtered[b].coords, filtered[b].coords + NUM_COORDS + 1);
        // mapping to original size
        denorm_detections(coords, scale, pad);
        for (size_t i = 0; i < NUM_COORDS + 1; ++i)
            filtered[b].coords[i] = coords[i];
    }

    return filtered;
}

cv::Mat draw_detections(cv::Mat image, const std::vector<BlazePoseDetection> &detections)
{
    cv::Mat drawn_image = image.clone();
    int class_id = 0;
    for (const auto &det : detections)
    {
        // Generate color based on class_id using HSV
        float hue = fmod(class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar color(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);

        // Draw bounding box
        int x1 = static_cast<int>(det.coords[1]);
        int y1 = static_cast<int>(det.coords[0]);
        int x2 = static_cast<int>(det.coords[3]);
        int y2 = static_cast<int>(det.coords[2]);
        cv::rectangle(drawn_image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // Draw label
        std::string label = std::string(SHOW_CLASSES[class_id]) + ": " + cv::format("%.2f", det.coords[12]);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        int label_x = x1;
        int label_y = y1 - 5;
        if (label_y < text_size.height)
            label_y = x1 + text_size.height + 5;

        // Draw label background
        cv::rectangle(drawn_image,
                      cv::Point(label_x, label_y - text_size.height - baseline),
                      cv::Point(label_x + text_size.width, label_y + baseline),
                      color, cv::FILLED);

        // Determine text color based on background brightness
        int brightness = (color[0] + color[1] + color[2]) / 3;
        cv::Scalar text_color = brightness < 128 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);

        cv::putText(drawn_image, label,
                    cv::Point(label_x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv::LINE_AA);
    }
    return drawn_image;
}

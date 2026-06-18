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
#include <fstream>
#include <sstream>
#include <algorithm>

static const std::vector<std::pair<int, int>> POSE_CONNECTIONS = {
    // Face
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 7},
    {0, 4},
    {4, 5},
    {5, 6},
    {6, 8},
    // Mouth
    {9, 10},
    // Shoulders
    {11, 12},
    // Right arm
    {11, 13},
    {13, 15},
    {15, 17},
    {15, 19},
    {15, 21},
    {17, 19},
    // Left arm
    {12, 14},
    {14, 16},
    {16, 18},
    {16, 20},
    {16, 22},
    {18, 20},
    // Torso
    {11, 23},
    {12, 24},
    {23, 24},
    // Right leg
    {23, 25},
    {25, 27},
    {27, 29},
    {27, 31},
    {29, 31},
    // Left leg
    {24, 26},
    {26, 28},
    {28, 30},
    {28, 32},
    {30, 32}};

#define LOGE(...)                     \
    do                                \
    {                                 \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

std::tuple<cv::Mat, ROI> preprocess(cv::Mat img, std::vector<std::vector<float>> &detections, std::tuple<int, int> new_shape)
{
    cv::Mat img_rgb;
    if (img.empty())
    {
        LOGE("Preprocess received empty image");
        return {};
    }

    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    if (detections.empty())
    {
        LOGE("No detections provided");
        return {};
    }

    auto &det = detections[0];
    float x_center = det[4];
    float y_center = det[5];
    float x_scale = det[6];
    float y_scale = det[7];

    float box_size = std::sqrt((x_scale - x_center) * (x_scale - x_center) + (y_scale - y_center) * (y_scale - y_center)) * 2.f;
    box_size *= 1.25f;

    float angle = (M_PI * 90.f / 180.f) - std::atan2(-(y_scale - y_center), x_scale - x_center);
    float rotation = angle - 2.f * M_PI * std::floor((angle - (-M_PI)) / (2.f * M_PI));

    cv::RotatedRect rotated_rect(cv::Point2f(x_center, y_center), cv::Size2f(box_size, box_size), rotation * 180.f / M_PI);
    cv::Point2f pts1[4];
    rotated_rect.points(pts1);

    int w = std::get<1>(new_shape);
    int h = std::get<0>(new_shape);

    cv::Point2f pts2[4] = {
        cv::Point2f(0.f, (float)h),
        cv::Point2f(0.f, 0.f),
        cv::Point2f((float)w, 0.f),
        cv::Point2f((float)w, (float)h)};

    cv::Mat M = cv::getPerspectiveTransform(pts1, pts2);
    cv::Mat processed_img;
    cv::warpPerspective(img_rgb, processed_img, M, cv::Size(w, h), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Mat img_float;
    processed_img.convertTo(img_float, CV_32F, 1.0 / 255.0);

    ROI roi = {x_center, y_center, box_size, rotation};

    return std::make_tuple(img_float, roi);
}

cv::Mat quantize_input(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
{
    // Type 0 is Float32 (No quantization needed)
    if (attr.type == 0)
        return float_img.clone();

    // Type 3 is UINT8, otherwise assume INT8
    int mat_type = (attr.type == 3) ? CV_8UC3 : CV_8SC3;
    cv::Mat quantized_img(float_img.rows, float_img.cols, mat_type);

    const float *src = (const float *)float_img.data;
    int total = float_img.total() * float_img.channels();

    if (attr.type == 3)
    { // UINT8
        uint8_t *dst = (uint8_t *)quantized_img.data;
        for (int i = 0; i < total; ++i)
        {
            float val = std::round(src[i] / attr.scale) + attr.zp;
            dst[i] = static_cast<uint8_t>(std::clamp(val, 0.f, 255.f));
        }
    }
    else
    { // INT8
        int8_t *dst = (int8_t *)quantized_img.data;
        for (int i = 0; i < total; ++i)
        {
            float val = std::round(src[i] / attr.scale) + attr.zp;
            dst[i] = static_cast<int8_t>(std::clamp(val, -128.f, 127.f));
        }
    }
    return quantized_img;
}

void refine_landmark(std::vector<std::vector<float>> &landmarks, const float *heatmap, int hm_w, int hm_h, int hm_c)
{
    float min_confidence = 0.5f;
    int kernel_size = 9;
    int offset = kernel_size;

    for (size_t i = 0; i < landmarks.size(); ++i)
    {
        int col = static_cast<int>(landmarks[i][0] * hm_w);
        int row = static_cast<int>(landmarks[i][1] * hm_h);

        if (!(col >= 0 && col < hm_w && row >= 0 && row < hm_h))
        {
            continue;
        }

        int c0 = std::max(0, col - offset);
        int c1 = std::min(hm_w, col + offset + 1);
        int r0 = std::max(0, row - offset);
        int r1 = std::min(hm_h, row + offset + 1);

        float val_sum = 0.0f;
        float weighted_col = 0.0f;
        float weighted_row = 0.0f;
        float max_conf = 0.0f;

        for (int r = r0; r < r1; ++r)
        {
            for (int c = c0; c < c1; ++c)
            {
                float val = heatmap[r * hm_w * hm_c + c * hm_c + i];
                float conf = sigmoid(val);
                val_sum += conf;
                max_conf = std::max(max_conf, conf);
                weighted_col += c * conf;
                weighted_row += r * conf;
            }
        }

        if (max_conf >= min_confidence && val_sum > 0)
        {
            landmarks[i][0] = weighted_col / (hm_w * val_sum);
            landmarks[i][1] = weighted_row / (hm_h * val_sum);
        }
    }
}

std::vector<BlazePoseLandmark> postprocess(float *raw_landmarks, float *raw_heatmap, const ROI &roi)
{
    std::vector<std::vector<float>> landmarks(NUM_LANDMARKS, std::vector<float>(5, 0.0f));
    for (int i = 0; i < NUM_LANDMARKS; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            float val = raw_landmarks[i * 5 + j];

            if (j == 3 || j == 4)
                val = sigmoid(val);
            if (j < 3)
                val = val / IMAGE_SIZE;

            landmarks[i][j] = val;
        }
    }

    // Refine landmarks
    if (raw_heatmap)
    {
        refine_landmark(landmarks, raw_heatmap, 64, 64, 39);
    }

    // Denormalize coordinates to Original Image Space
    float cosa = std::cos(roi.rotation);
    float sina = std::sin(roi.rotation);

    std::vector<BlazePoseLandmark> pose_res;
    BlazePoseLandmark pose;
    pose.landmarks.resize(NUM_LANDMARKS);

    for (int i = 0; i < NUM_LANDMARKS; ++i)
    {
        float x = landmarks[i][0] - 0.5f;
        float y = landmarks[i][1] - 0.5f;
        float z = landmarks[i][2];

        float new_x = (cosa * x - sina * y) * roi.box_size + roi.x_center;
        float new_y = (sina * x + cosa * y) * roi.box_size + roi.y_center;
        float new_z = z * roi.box_size;

        float score = landmarks[i][3]; // Use visibility as score

        pose.landmarks[i] = {(double)new_x, (double)new_y, (double)new_z, (double)score};
    }

    pose_res.push_back(pose);
    return pose_res;
}

std::vector<std::vector<float>> load_detections(const std::string &txt_path)
{
    std::vector<std::vector<float>> detections;
    std::ifstream ifs(txt_path);

    for (std::string line; std::getline(ifs, line);)
    {
        std::istringstream iss(line);
        std::vector<float> det;
        float val;

        while (iss >> val)
        {
            det.push_back(val);
        }

        if (!det.empty())
        {
            detections.push_back(det);
        }
    }

    return detections;
}

cv::Mat draw_landmarks(cv::Mat image, const std::vector<BlazePoseLandmark> &landmarks, float score_threshold)
{
    cv::Mat out = image.clone();

    for (const auto &lm : landmarks)
    {
        const auto &lms = lm.landmarks;

        for (size_t i = 0; i < lms.size(); ++i)
        {
            int x = static_cast<int>(lms[i][0]);
            int y = static_cast<int>(lms[i][1]);
            double v = lms[i][3];

            if (v < score_threshold)
                continue;

            cv::circle(out, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        }

        for (const auto &conn : POSE_CONNECTIONS)
        {
            int i0 = conn.first;
            int i1 = conn.second;

            if (i0 >= lms.size() || i1 >= lms.size())
                continue;

            if (lms[i0][3] < score_threshold || lms[i1][3] < score_threshold)
                continue;

            cv::Point p0(static_cast<int>(lms[i0][0]), static_cast<int>(lms[i0][1]));
            cv::Point p1(static_cast<int>(lms[i1][0]), static_cast<int>(lms[i1][1]));

            cv::line(out, p0, p1, cv::Scalar(255, 0, 0), 2);
        }
    }

    return out;
}
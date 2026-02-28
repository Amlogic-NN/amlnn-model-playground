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
const char *SHOW_CLASSES[1] = {"lm"};

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

struct ROI
{
    float x_center;
    float y_center;
    float scale;
    float theta;
};

ROI detection_to_roi(const std::vector<float> &detection, int kp1 = 0, int kp2 = 1)
{
    float theta0 = 90.f * M_PI / 180.f;
    float dscale = 1.1f; // 1.0 * 256 / 224; // 1.1f;
    float dy = 0.f;

    float x_center = detection[4 + 2 * kp1];
    float y_center = detection[4 + 2 * kp1 + 1];
    float x1 = detection[4 + 2 * kp2];
    float y1 = detection[4 + 2 * kp2 + 1];

    float roi_scale = std::sqrt((x_center - x1) * (x_center - x1) + (y_center - y1) * (y_center - y1)) * 2.f;
    y_center += dy * roi_scale;
    roi_scale *= dscale;

    float theta = std::atan2(detection[4 + 2 * kp1 + 1] - detection[4 + 2 * kp2 + 1], detection[4 + 2 * kp1] - detection[4 + 2 * kp2]) - theta0;

    return {x_center, y_center, roi_scale, theta};
}

cv::Mat extract_roi(cv::Mat &frame, const ROI &roi, int resolution, cv::Mat &affine)
{
    cv::Point2f src_pts[3];
    src_pts[0] = cv::Point2f(-roi.scale / 2.f, -roi.scale / 2.f); // will map to (0,0)
    src_pts[1] = cv::Point2f(-roi.scale / 2.f, roi.scale / 2.f);  // will map to (0,res-1)
    src_pts[2] = cv::Point2f(roi.scale / 2.f, -roi.scale / 2.f);  // will map to (res-1,0)
    float cos_theta = std::cos(roi.theta);
    float sin_theta = std::sin(roi.theta);
    for (int i = 0; i < 3; i++)
    {
        float x = src_pts[i].x;
        float y = src_pts[i].y;
        src_pts[i].x = roi.x_center + x * cos_theta - y * sin_theta;
        src_pts[i].y = roi.y_center + x * sin_theta + y * cos_theta;
    }
    cv::Point2f dst_pts[3] = {
        cv::Point2f(0.f, 0.f),
        cv::Point2f(0.f, resolution - 1.f),
        cv::Point2f(resolution - 1.f, 0.f)};

    cv::Mat roi_img;
    cv::Mat M = cv::getAffineTransform(src_pts, dst_pts);
    cv::invertAffineTransform(M, affine);
    cv::warpAffine(frame, roi_img, M, cv::Size(resolution, resolution));

    return roi_img;
}

std::tuple<cv::Mat, cv::Mat> preprocess(cv::Mat img, std::vector<std::vector<float>> &detections, std::tuple<int, int> new_shape)
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

    ROI roi = detection_to_roi(detections[0]); // get the first bounding box
    cv::Mat affine;
    cv::Mat roi_img = extract_roi(img_rgb, roi, IMAGE_SIZE, affine);

    cv::Mat img_float;
    roi_img.convertTo(img_float, CV_32F, 1.0 / 255.0);

    return std::make_tuple(img_float, affine);
}

cv::Mat quantize_input(const cv::Mat &float_img, float scale, int16_t zero_point)
{
    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        LOGE("quantize_input: Invalid input image (must be CV_32FC3)");
        return cv::Mat();
    }

    cv::Mat quantized_img(float_img.rows, float_img.cols, CV_16SC3);
    const float *src_ptr = (const float *)float_img.data;
    int16_t *dst_ptr = (int16_t *)quantized_img.data;

    int total_elements = float_img.total() * float_img.channels();
    // for (int i = 0; i < total_elements; ++i)
    // {
    //     dst_ptr[i] = static_cast<int16_t>(std::round(src_ptr[i] / scale + zero_point));
    // }
    for (int i = 0; i < total_elements; ++i)
    {
        int32_t q = static_cast<int32_t>(std::round(src_ptr[i] / scale));
        q = std::max(-32768, std::min(32767, q));
        dst_ptr[i] = static_cast<int16_t>(q);
    }

    return quantized_img;
}

void blazepose_postprocess(const float *landmarks, float *normalized_landmarks)
{
    if (!landmarks || !normalized_landmarks)
        return;

    for (int j = 0; j < NUM_LANDMARKS; j++)
    {
        float x = landmarks[j * LANDMARK_FEATURE_DIM + 0] / IMAGE_SIZE;
        float y = landmarks[j * LANDMARK_FEATURE_DIM + 1] / IMAGE_SIZE;
        float z = landmarks[j * LANDMARK_FEATURE_DIM + 2] / IMAGE_SIZE;
        float visibility = landmarks[j * LANDMARK_FEATURE_DIM + 3];
        float presence = landmarks[j * LANDMARK_FEATURE_DIM + 4];

        float score = sigmoid(fminf(visibility, presence));
        normalized_landmarks[j * LANDMARK_OUT_DIM + 0] = x;
        normalized_landmarks[j * LANDMARK_OUT_DIM + 1] = y;
        normalized_landmarks[j * LANDMARK_OUT_DIM + 2] = z;
        normalized_landmarks[j * LANDMARK_OUT_DIM + 3] = score;
    }
}

/**
 * Denormalize landmarks: map normalized coordinates back to original image using affine
 * @param landmarks   Input/Output: [NUM_LANDMARKS * LANDMARK_OUT_DIM], first three dimensions are x, y, z
 * @param affine      Input: [2 x 3] affine matrix (CV_32F)
 */
void blazepose_denorm_landmarks(float *landmarks, const cv::Mat &affine)
{
    if (!landmarks || affine.empty() || affine.rows != 2 || affine.cols != 3)
    {
        return;
    }

    const double *a = affine.ptr<double>();
    double a00 = a[0], a01 = a[1], a02 = a[2];
    double a10 = a[3], a11 = a[4], a12 = a[5];
    for (int j = 0; j < NUM_LANDMARKS; j++)
    {
        float *p = landmarks + j * LANDMARK_OUT_DIM;
        // scale to input resolution
        float x = p[0] * IMAGE_SIZE;
        float y = p[1] * IMAGE_SIZE;
        float z = p[2] * IMAGE_SIZE;

        // apply affine transform
        float new_x = a00 * x + a01 * y + a02;
        float new_y = a10 * x + a11 * y + a12;

        p[0] = new_x;
        p[1] = new_y;
        p[2] = z;

    }
}

std::vector<BlazePoseLandmark> postprocess(nn_output *outdata, const cv::Mat &affine)
{
    // keep all outputs, even if unused
    float *world_landmarks = (float *)outdata->out[0].buf;
    float *heatmap = (float *)outdata->out[1].buf;
    float *flags = (float *)outdata->out[2].buf;
    float *landmarks = (float *)outdata->out[4].buf;

    float *normalized_landmarks =
        new float[NUM_LANDMARKS * LANDMARK_OUT_DIM]();

    blazepose_postprocess(landmarks, normalized_landmarks);

    // refine_landmark_from_heatmap(normalized_landmarks, 39, heatmap, 64, 64);

    blazepose_denorm_landmarks(normalized_landmarks, affine);

    std::vector<BlazePoseLandmark> pose_res;
    pose_res.reserve(1);

    BlazePoseLandmark pose;
    pose.landmarks.resize(NUM_LANDMARKS);

    for (int i = 0; i < NUM_LANDMARKS; ++i)
    {
        int base = i * LANDMARK_OUT_DIM;

        double x = normalized_landmarks[base + 0];     // x
        double y = normalized_landmarks[base + 1];     // y
        double z = normalized_landmarks[base + 2];     // z
        double score = normalized_landmarks[base + 3]; // score

        pose.landmarks[i] = {x, y, z, score};
    }

    pose_res.push_back(pose);

    delete[] normalized_landmarks;
    normalized_landmarks = nullptr;

    return pose_res;
}

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

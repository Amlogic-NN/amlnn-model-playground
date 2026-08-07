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
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

static std::vector<std::array<float, 4>> generate_priors(int target_w, int target_h)
{
    std::vector<std::array<float, 4>> priors;
    std::vector<int> steps = {8, 16, 32};
    std::vector<std::vector<int>> min_sizes = {{16, 32}, {64, 128}, {256, 512}};

    for (size_t k = 0; k < steps.size(); ++k)
    {
        int fm_h = std::ceil((float)target_h / steps[k]);
        int fm_w = std::ceil((float)target_w / steps[k]);
        for (int i = 0; i < fm_h; i++)
        {
            for (int j = 0; j < fm_w; j++)
            {
                for (int ms : min_sizes[k])
                {
                    float cx = (j + 0.5f) * steps[k] / target_w;
                    float cy = (i + 0.5f) * steps[k] / target_h;
                    float sx = (float)ms / target_w;
                    float sy = (float)ms / target_h;
                    priors.push_back({cx, cy, sx, sy});
                }
            }
        }
    }
    return priors;
}

static std::array<float, 10> decode_landm(const float *lm, int idx, int total, bool is_planar, const std::array<float, 4> &p)
{
    std::array<float, 10> out{};
    float raw[10];
    if (is_planar)
    {
        for (int j = 0; j < 10; ++j)
            raw[j] = lm[j * total + idx];
    }
    else
    {
        for (int j = 0; j < 10; ++j)
            raw[j] = lm[idx * 10 + j];
    }
    for (int i = 0; i < 5; i++)
    {
        out[2 * i] = p[0] + raw[2 * i] * 0.1f * p[2];
        out[2 * i + 1] = p[1] + raw[2 * i + 1] * 0.1f * p[3];
    }
    return out;
}

static std::array<float, 4> decode_box(const float *loc, int idx, int total, bool is_planar, const std::array<float, 4> &p)
{
    float l[4];
    if (is_planar)
    {
        l[0] = loc[0 * total + idx];
        l[1] = loc[1 * total + idx];
        l[2] = loc[2 * total + idx];
        l[3] = loc[3 * total + idx];
    }
    else
    {
        l[0] = loc[idx * 4 + 0];
        l[1] = loc[idx * 4 + 1];
        l[2] = loc[idx * 4 + 2];
        l[3] = loc[idx * 4 + 3];
    }
    float cx = p[0] + l[0] * 0.1f * p[2];
    float cy = p[1] + l[1] * 0.1f * p[3];
    float w = p[2] * std::exp(l[2] * 0.2f);
    float h = p[3] * std::exp(l[3] * 0.2f);
    return {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f}; // returns x1, y1, x2, y2
}

static float compute_iou(const FaceDetection &a, const FaceDetection &b)
{
    float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter);
}

static std::vector<FaceDetection> apply_nms(std::vector<FaceDetection> &boxes, float thresh)
{
    std::sort(boxes.begin(), boxes.end(), [](const FaceDetection &a, const FaceDetection &b)
              { return a.score > b.score; });

    std::vector<FaceDetection> keep;
    std::vector<bool> removed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (removed[i])
            continue;
        keep.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (removed[j])
                continue;
            if (compute_iou(boxes[i], boxes[j]) > thresh)
            {
                removed[j] = true;
            }
        }
    }
    return keep;
}

int get_num_priors(int target_w, int target_h)
{
    static int num_priors = generate_priors(target_w, target_h).size();
    return num_priors;
}

std::tuple<cv::Mat, float, int, int> preprocess(const cv::Mat &img, int target_w, int target_h)
{
    if (img.empty())
        return {cv::Mat(), 0.0f, 0, 0};

    int h0 = img.rows;
    int w0 = img.cols;

    float scale = std::min((float)target_w / w0, (float)target_h / h0);
    int nw = static_cast<int>(w0 * scale);
    int nh = static_cast<int>(h0 * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));

    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(128, 128, 128));
    int pad_x = (target_w - nw) / 2;
    int pad_y = (target_h - nh) / 2;
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, nw, nh)));

    cv::Mat float_img(target_h, target_w, CV_32FC3);
    float mean[3] = {104.0f, 117.0f, 123.0f};

    for (int i = 0; i < target_h; ++i)
    {
        for (int j = 0; j < target_w; ++j)
        {
            cv::Vec3b pixel = canvas.at<cv::Vec3b>(i, j);
            float_img.at<cv::Vec3f>(i, j)[0] = pixel[0] - mean[0];
            float_img.at<cv::Vec3f>(i, j)[1] = pixel[1] - mean[1];
            float_img.at<cv::Vec3f>(i, j)[2] = pixel[2] - mean[2];
        }
    }

    return {float_img, scale, pad_x, pad_y};
}

cv::Mat quantize_input(const cv::Mat &float_img, float scale, int32_t zero_point, int tensor_type)
{
    cv::Mat flat_img = float_img.isContinuous() ? float_img : float_img.clone();
    int total_elements = flat_img.total() * flat_img.channels();
    const float *src = (const float *)flat_img.data;

    if (scale < 1e-6f)
        scale = 1.0f;

    cv::Mat quantized_img(1, total_elements, (tensor_type == 3) ? CV_8UC1 : CV_8SC1);

    if (tensor_type == 3)
    {
        uint8_t *dst = (uint8_t *)quantized_img.data;
        for (int i = 0; i < total_elements; ++i)
        {
            dst[i] = static_cast<uint8_t>(std::clamp(std::nearbyint(src[i] / scale) + zero_point, 0.0f, 255.0f));
        }
    }
    else
    {
        int8_t *dst = (int8_t *)quantized_img.data;
        for (int i = 0; i < total_elements; ++i)
        {
            dst[i] = static_cast<int8_t>(std::clamp(std::nearbyint(src[i] / scale) + zero_point, -128.0f, 127.0f));
        }
    }
    return quantized_img;
}

std::vector<FaceDetection> postprocess(float *loc, bool loc_planar,
                                       float *conf, bool conf_planar,
                                       float *landm, bool landm_planar,
                                       std::tuple<cv::Mat, float, int, int> input_tuple,
                                       int target_w, int target_h,
                                       float conf_thresh, float nms_thresh)
{
    float scale = std::get<1>(input_tuple);
    int pad_x = std::get<2>(input_tuple);
    int pad_y = std::get<3>(input_tuple);

    // Static initialization so priors are only calculated once
    static const auto priors = generate_priors(target_w, target_h);
    int num_priors = priors.size();

    std::vector<FaceDetection> candidates;

    for (int j = 0; j < num_priors; j++)
    {
        float score = conf_planar ? conf[num_priors + j] : conf[j * 2 + 1];

        if (score >= conf_thresh)
        {
            auto b = decode_box(loc, j, num_priors, loc_planar, priors[j]);
            auto lm = decode_landm(landm, j, num_priors, landm_planar, priors[j]);

            FaceDetection det;
            det.score = score;

            // Reverse Letterbox Mapping (normalized -> pixel coordinates -> orig image)
            det.x1 = std::max(0.0f, (b[0] * target_w - pad_x) / scale);
            det.y1 = std::max(0.0f, (b[1] * target_h - pad_y) / scale);
            det.x2 = std::max(0.0f, (b[2] * target_w - pad_x) / scale);
            det.y2 = std::max(0.0f, (b[3] * target_h - pad_y) / scale);

            for (int i = 0; i < 5; i++)
            {
                det.landmarks[2 * i] = (lm[2 * i] * target_w - pad_x) / scale;
                det.landmarks[2 * i + 1] = (lm[2 * i + 1] * target_h - pad_y) / scale;
            }

            candidates.push_back(det);
        }
    }

    return apply_nms(candidates, nms_thresh);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<FaceDetection> &detections)
{
    cv::Mat drawn_image = image.clone();

    int face_id = 1;
    for (const auto &det : detections)
    {
        // Draw Rectangle
        cv::rectangle(drawn_image,
                      cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                      cv::Scalar(0, 255, 0), 2);

        // Draw Score Label
        std::string label = "Face " + std::to_string(face_id++) + " (" + cv::format("%.2f", det.score) + ")";
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::putText(drawn_image, label,
                    cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        // Draw 5 Landmarks
        for (int j = 0; j < 5; j++)
        {
            cv::circle(drawn_image,
                       cv::Point(static_cast<int>(det.landmarks[2 * j]), static_cast<int>(det.landmarks[2 * j + 1])),
                       2, cv::Scalar(0, 0, 255), -1);
        }
    }
    return drawn_image;
}
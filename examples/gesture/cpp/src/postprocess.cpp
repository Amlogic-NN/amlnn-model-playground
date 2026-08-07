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
#include <unordered_map>
#include <opencv2/opencv.hpp>

#define LOGE(...)                     \
    do                                \
    {                                 \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

const char *NAMES[19] = {
    "ok", "stop", "palm", "like", "dislike", "no_gesture", "call", "fist",
    "four", "mute", "one", "peace", "peace_inverted", "rock",
    "stop_inverted", "three", "three2", "two_up", "two_up_inverted"};

const float STRIDES[3] = {32.0f, 16.0f, 8.0f};
const int GRIDS[3] = {20, 40, 80};
const float ANCHOR_GRIDS[3][3][2] = {
    {{116, 90}, {156, 198}, {373, 326}}, // Grid 20
    {{30, 61}, {62, 45}, {59, 119}},     // Grid 40
    {{10, 13}, {16, 30}, {33, 23}}       // Grid 80
};

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
        {
            shape.push_back(attr.dims[i]);
        }
    }
    return shape;
}

static float compute_iou(const GestureDetection &d1, const GestureDetection &d2)
{
    float xx1 = std::max(d1.x1, d2.x1);
    float yy1 = std::max(d1.y1, d2.y1);
    float xx2 = std::min(d1.x2, d2.x2);
    float yy2 = std::min(d1.y2, d2.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area1 = std::max(0.0f, d1.x2 - d1.x1) * std::max(0.0f, d1.y2 - d1.y1);
    float area2 = std::max(0.0f, d2.x2 - d2.x1) * std::max(0.0f, d2.y2 - d2.y1);

    return inter / std::max(area1 + area2 - inter, 1e-6f);
}

std::tuple<cv::Mat, int, int> preprocess(const cv::Mat &img, int input_size)
{
    if (img.empty())
    {
        LOGE("Preprocess received empty image");
        return {cv::Mat(), 0, 0};
    }

    int orig_w = img.cols;
    int orig_h = img.rows;

    cv::Mat rgb, resized, float_img;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(input_size, input_size));

    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    return {float_img, orig_w, orig_h};
}

std::vector<GestureDetection> postprocess(
    const std::vector<float *> &out_data,
    const std::vector<std::vector<int>> &out_shapes,
    const std::tuple<cv::Mat, int, int> &prep_info,
    float conf_thresh,
    float nms_thresh)
{
    int orig_w = std::get<1>(prep_info);
    int orig_h = std::get<2>(prep_info);
    int input_size = std::get<0>(prep_info).cols;

    std::vector<GestureDetection> candidates;

    for (size_t i = 0; i < out_data.size(); ++i)
    {
        float *data = out_data[i];
        const auto &shape = out_shapes[i];

        // Safely determine grid size based on the spatial dimension (400, 1600, or 6400)
        int g = 0;
        for (int dim : shape)
        {
            if (dim == 400)
            {
                g = 20;
                break;
            }
            if (dim == 1600)
            {
                g = 40;
                break;
            }
            if (dim == 6400)
            {
                g = 80;
                break;
            }
        }

        // If we didn't find a matching grid, skip this output
        if (g == 0)
            continue;

        int grid_idx = (g == 20) ? 0 : (g == 40) ? 1
                                                 : 2;
        float stride = STRIDES[grid_idx];

        for (int h = 0; h < g; ++h)
        {
            for (int w = 0; w < g; ++w)
            {
                int spatial_idx = h * g + w; // 0 to g*g - 1

                for (int a = 0; a < 3; ++a)
                {
                    // The memory layout is [batch(1), g*g, 24, 3].
                    // 1. Get Objectness score (c = 4)
                    float obj = data[spatial_idx * 72 + 4 * 3 + a];

                    // 2. Find max Class score (c = 5 to 23)
                    float max_cls = -1.0f;
                    int cls_id = -1;
                    for (int c = 5; c < 24; ++c)
                    {
                        float cls_val = data[spatial_idx * 72 + c * 3 + a];
                        if (cls_val > max_cls)
                        {
                            max_cls = cls_val;
                            cls_id = c - 5;
                        }
                    }

                    // 3. Compute final score
                    float score = obj * max_cls;
                    if (score < conf_thresh)
                        continue;

                    // 4. Get Bounding Box params (c = 0, 1, 2, 3)
                    float tx = data[spatial_idx * 72 + 0 * 3 + a];
                    float ty = data[spatial_idx * 72 + 1 * 3 + a];
                    float tw = data[spatial_idx * 72 + 2 * 3 + a];
                    float th = data[spatial_idx * 72 + 3 * 3 + a];

                    // 5. Decode
                    float px = (tx * 2.0f - 0.5f + w) * stride;
                    float py = (ty * 2.0f - 0.5f + h) * stride;
                    float pw = (tw * 2.0f) * (tw * 2.0f) * ANCHOR_GRIDS[grid_idx][a][0];
                    float ph = (th * 2.0f) * (th * 2.0f) * ANCHOR_GRIDS[grid_idx][a][1];

                    float x1 = px - pw / 2.0f;
                    float y1 = py - ph / 2.0f;
                    float x2 = px + pw / 2.0f;
                    float y2 = py + ph / 2.0f;

                    candidates.push_back({x1, y1, x2, y2, score, cls_id});
                }
            }
        }
    }

    // Per-class NMS
    std::vector<GestureDetection> kept;
    std::unordered_map<int, std::vector<GestureDetection>> class_map;
    for (const auto &c : candidates)
    {
        class_map[c.class_id].push_back(c);
    }

    for (auto &[cls_id, cls_dets] : class_map)
    {
        std::sort(cls_dets.begin(), cls_dets.end(), [](const GestureDetection &a, const GestureDetection &b)
                  { return a.score > b.score; });

        std::vector<bool> removed(cls_dets.size(), false);
        for (size_t i = 0; i < cls_dets.size(); ++i)
        {
            if (removed[i])
                continue;
            kept.push_back(cls_dets[i]);

            for (size_t j = i + 1; j < cls_dets.size(); ++j)
            {
                if (removed[j])
                    continue;
                if (compute_iou(cls_dets[i], cls_dets[j]) > nms_thresh)
                {
                    removed[j] = true;
                }
            }
        }
    }

    // Scale boxes back to original image dimensions
    float scale_x = orig_w / static_cast<float>(input_size);
    float scale_y = orig_h / static_cast<float>(input_size);

    for (auto &d : kept)
    {
        d.x1 = std::max(0.0f, std::min(static_cast<float>(orig_w - 1), d.x1 * scale_x));
        d.y1 = std::max(0.0f, std::min(static_cast<float>(orig_h - 1), d.y1 * scale_y));
        d.x2 = std::max(0.0f, std::min(static_cast<float>(orig_w - 1), d.x2 * scale_x));
        d.y2 = std::max(0.0f, std::min(static_cast<float>(orig_h - 1), d.y2 * scale_y));
    }

    // Sort globally by score
    std::sort(kept.begin(), kept.end(), [](const GestureDetection &a, const GestureDetection &b)
              { return a.score > b.score; });

    return kept;
}

cv::Mat draw_detections(const cv::Mat &bgr, const std::vector<GestureDetection> &detections)
{
    cv::Mat vis = bgr.clone();
    int h = vis.rows;
    int w = vis.cols;

    double font_scale = std::max(0.8, std::min(w, h) / 600.0);
    int font_thickness = std::max(2, static_cast<int>(std::min(w, h) / 300));
    int box_thickness = std::max(2, static_cast<int>(std::min(w, h) / 250));

    for (const auto &det : detections)
    {
        std::string label = std::string(NAMES[det.class_id]) + " " + cv::format("%.2f", det.score);
        cv::rectangle(vis, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), box_thickness);

        int text_y = std::max(30, static_cast<int>(det.y1 - 10));
        cv::putText(vis, label, cv::Point(det.x1, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), font_thickness, cv::LINE_AA);
    }

    return vis;
}
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
#include <cmath>
#include <algorithm>
#include <iostream>

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

std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr)
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

static float compute_iou(const std::vector<float> &box1, const std::vector<float> &box2)
{
    float xx1 = std::max(box1[0], box2[0]);
    float yy1 = std::max(box1[1], box2[1]);
    float xx2 = std::min(box1[2], box2[2]);
    float yy2 = std::min(box1[3], box2[3]);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    float union_area = area1 + area2 - inter;
    return inter / std::max(union_area, 1e-6f);
}

std::tuple<cv::Mat, float, float> preprocess(const cv::Mat &img, int input_width, int input_height)
{
    if (img.empty())
    {
        LOGE("Preprocess received empty image");
        return {};
    }

    // Direct resize (matching Python script, no letterbox)
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    // Normalize to [0, 1]
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32F, 1.0 / 255.0);

    float sx = static_cast<float>(img.cols) / input_width;
    float sy = static_cast<float>(img.rows) / input_height;

    return std::make_tuple(img_float, sx, sy);
}

std::vector<Detection> postprocess(float *out_data, const std::vector<int> &out_shape,
                                          float sx, float sy, int orig_w, int orig_h,
                                          float conf_thresh, float iou_thresh, int pad)
{
    int num_anchors = 0;
    bool channels_last = false;

    // Detect format from shape dims
    if (out_shape.size() >= 2)
    {
        if (out_shape[0] == 5)
        {
            num_anchors = out_shape[1];
            channels_last = false;
        }
        else if (out_shape[1] == 5)
        {
            num_anchors = out_shape[0];
            channels_last = true;
        }
        else
        {
            num_anchors = out_shape[out_shape.size() - 2];
        }
    }

    struct Candidate
    {
        std::vector<float> box_320;
        float score;
    };
    std::vector<Candidate> candidates;

    for (int i = 0; i < num_anchors; ++i)
    {
        int idx_x = channels_last ? (i * 5 + 0) : (0 * num_anchors + i);
        int idx_y = channels_last ? (i * 5 + 1) : (1 * num_anchors + i);
        int idx_w = channels_last ? (i * 5 + 2) : (2 * num_anchors + i);
        int idx_h = channels_last ? (i * 5 + 3) : (3 * num_anchors + i);
        int idx_s = channels_last ? (i * 5 + 4) : (4 * num_anchors + i);

        float raw_score = out_data[idx_s];
        float score = sigmoid(raw_score);

        if (score >= conf_thresh)
        {
            float cx = out_data[idx_x];
            float cy = out_data[idx_y];
            float w = out_data[idx_w];
            float h = out_data[idx_h];

            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            candidates.push_back({{x1, y1, x2, y2}, score});
        }
    }

    // Sort by score
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b)
              { return a.score > b.score; });

    // NMS
    std::vector<bool> removed(candidates.size(), false);
    std::vector<Detection> results;

    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (removed[i])
            continue;

        float x1_320 = candidates[i].box_320[0];
        float y1_320 = candidates[i].box_320[1];
        float x2_320 = candidates[i].box_320[2];
        float y2_320 = candidates[i].box_320[3];

        // Scale coordinates back
        float x1 = x1_320 * sx;
        float y1 = y1_320 * sy;
        float x2 = x2_320 * sx;
        float y2 = y2_320 * sy;

        // Add padding identically to Python script
        int x1p = std::max(0, static_cast<int>(x1 - pad - 10));
        int y1p = std::max(0, static_cast<int>(y1 - pad - 15));
        int x2p = std::min(orig_w - 1, static_cast<int>(x2 + pad - 10));
        int y2p = std::min(orig_h - 1, static_cast<int>(y2 + pad));

        results.push_back({x1p, y1p, x2p, y2p, candidates[i].score, "", {}});

        for (size_t j = i + 1; j < candidates.size(); ++j)
        {
            if (removed[j])
                continue;
            if (compute_iou(candidates[i].box_320, candidates[j].box_320) > iou_thresh)
            {
                removed[j] = true;
            }
        }
    }

    return results;
}

std::vector<Detection> decode(const cv::Mat &orig_img, const std::vector<Detection> &detections)
{
    cv::QRCodeDetector detector;
    std::vector<Detection> decoded_results;

    int orig_h = orig_img.rows;
    int orig_w = orig_img.cols;

    for (const auto &det : detections)
    {
        int x1 = std::max(0, std::min(det.x1, orig_w - 1));
        int y1 = std::max(0, std::min(det.y1, orig_h - 1));
        int x2 = std::max(0, std::min(det.x2, orig_w - 1));
        int y2 = std::max(0, std::min(det.y2, orig_h - 1));

        if (x2 <= x1 || y2 <= y1)
        {
            std::cout << "skip invalid box: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]\n";
            continue;
        }

        cv::Mat crop = orig_img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        if (crop.empty())
            continue;

        int cw = crop.cols;
        int ch = crop.rows;
        if (cw < 20 || ch < 20)
        {
            std::cout << "skip too small crop: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "], size=(" << cw << "," << ch << ")\n";
            continue;
        }

        Detection out_det = det;
        out_det.x1 = x1;
        out_det.y1 = y1;
        out_det.x2 = x2;
        out_det.y2 = y2;

        try
        {
            std::vector<cv::Point> points;
            cv::Mat straight_qrcode;
            std::string text = detector.detectAndDecode(crop, points, straight_qrcode);

            out_det.text = text;
            out_det.points = points;
            decoded_results.push_back(out_det);
        }
        catch (const cv::Exception &e)
        {
            std::cout << "skip cv2 decode error: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "], err=" << e.what() << "\n";
        }
    }

    return decoded_results;
}

cv::Mat draw_results(const cv::Mat &image, const std::vector<Detection> &detections)
{
    cv::Mat vis = image.clone();

    for (const auto &r : detections)
    {
        cv::rectangle(vis, cv::Point(r.x1, r.y1), cv::Point(r.x2, r.y2), cv::Scalar(0, 255, 0), 2);

        char score_text[32];
        snprintf(score_text, sizeof(score_text), "%.3f", r.score);
        cv::putText(vis, score_text, cv::Point(r.x1, std::max(0, r.y1 - 8)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        if (!r.text.empty())
        {
            std::string text_disp = r.text.length() > 60 ? r.text.substr(0, 60) : r.text;
            cv::putText(vis, text_disp, cv::Point(r.x1, std::min(image.rows - 10, r.y2 + 25)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        }
    }

    return vis;
}
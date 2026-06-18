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

#define LOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)

const std::vector<std::string> KEYPOINT_NAMES = {
    "nose","l_eye","r_eye","l_ear","r_ear",
    "l_sh","r_sh","l_el","r_el","l_wr","r_wr",
    "l_hip","r_hip","l_kn","r_kn","l_an","r_an"
};

const std::vector<std::pair<int, int>> SKELETON = {
    {0,1}, {0,2}, {1,3}, {2,4}, {5,6}, {5,7}, {7,9}, {6,8},
    {8,10}, {5,11}, {6,12}, {11,12}, {11,13}, {13,15}, {12,14}, {14,16}
};

// Helper function to extract meaningful dimensions (ignores batch dim 1)
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

static float compute_iou(const Detection& det1, const Detection& det2) {
    float xx1 = std::max(det1.x1, det2.x1);
    float yy1 = std::max(det1.y1, det2.y1);
    float xx2 = std::min(det1.x2, det2.x2);
    float yy2 = std::min(det1.y2, det2.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1);
    float area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1);

    return inter / (area1 + area2 - inter);
}

static std::vector<Detection> nms(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return {};

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<Detection> final_detections;
    std::vector<bool> removed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (removed[i]) continue;
        final_detections.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (removed[j]) continue;
            if (compute_iou(detections[i], detections[j]) > iou_threshold) {
                removed[j] = true;
            }
        }
    }
    return final_detections;
}

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape) {
    cv::Mat img_rgb;
    if (img.empty()) { LOGE("Preprocess received empty image"); return {}; }

    if (img.channels() == 4) cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3) cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else img_rgb = img.clone();

    int orig_h = img.rows, orig_w = img.cols;
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
    cv::copyMakeBorder(img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32F, 1.0 / 255.0);

    return std::make_tuple(img_float, scale, std::make_tuple(pad_left, pad_top));
}

cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point) {
    if (float_img.empty() || float_img.type() != CV_32FC3) return cv::Mat();

    cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8SC3);
    const float* src_ptr = (const float*)float_img.data;
    int8_t* dst_ptr = (int8_t*)quantized_img.data;

    int total_elements = float_img.total() * float_img.channels();
    for (int i = 0; i < total_elements; ++i) {
        float val = std::round(src_ptr[i] / scale) + zero_point;
        dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
    }
    return quantized_img;
}

// --- Layout Independent Memory Parsers ---
std::vector<int> get_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

int get_idx_from_shape(const std::vector<int>& shape, const std::vector<int>& strides, int val_c, int val_n, int val_k) {
    int offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 100) offset += val_n * strides[i]; // N dimension (~8400)
        else if (shape[i] == 17) offset += val_k * strides[i]; // Kpts dimension
        else if (shape[i] <= 4) offset += val_c * strides[i]; // Ch dimension (2 or 4)
    }
    return offset;
}

std::vector<Detection> postprocess(float* bbox_data, const std::vector<int>& bbox_shape,
                                   float* score_data, const std::vector<int>& score_shape,
                                   float* kpt_conf_data, const std::vector<int>& kpt_conf_shape,
                                   float* kpt_xy_data, const std::vector<int>& kpt_xy_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    int num_anchors = 8400;
    for (int d : score_shape) { if (d > 100) { num_anchors = d; break; } }

    std::vector<int> stride_bbox = get_strides(bbox_shape);
    std::vector<int> stride_score = get_strides(score_shape);
    std::vector<int> stride_conf = get_strides(kpt_conf_shape);
    std::vector<int> stride_xy = get_strides(kpt_xy_shape);

    std::vector<Detection> detections_orig;

    for (int i = 0; i < num_anchors; ++i) {
        float score = score_data[get_idx_from_shape(score_shape, stride_score, 0, i, 0)];

        if (score >= conf_thresh) {
            float cx = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 0, i, 0)];
            float cy = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 1, i, 0)];
            float w  = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 2, i, 0)];
            float h  = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 3, i, 0)];

            float x1 = (cx - w / 2.0f - pad_left) / scale;
            float y1 = (cy - h / 2.0f - pad_top) / scale;
            float x2 = (cx + w / 2.0f - pad_left) / scale;
            float y2 = (cy + h / 2.0f - pad_top) / scale;

            Detection det;
            det.x1 = std::max(0.0f, x1);
            det.y1 = std::max(0.0f, y1);
            det.x2 = std::max(0.0f, x2);
            det.y2 = std::max(0.0f, y2);
            det.score = score;

            for (int k = 0; k < 17; ++k) {
                float conf = kpt_conf_data[get_idx_from_shape(kpt_conf_shape, stride_conf, 0, i, k)];
                float kx = kpt_xy_data[get_idx_from_shape(kpt_xy_shape, stride_xy, 0, i, k)];
                float ky = kpt_xy_data[get_idx_from_shape(kpt_xy_shape, stride_xy, 1, i, k)];

                kx = (kx - pad_left) / scale;
                ky = (ky - pad_top) / scale;

                det.kpt_conf.push_back(conf);
                det.keypoints.push_back({kx, ky});
            }

            detections_orig.push_back(det);
        }
    }

    return nms(detections_orig, iou_threshold);
}

cv::Scalar get_color(float conf) {
    float hue = fmod(conf * 137.508f, 360.0f);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections) {
    cv::Mat drawn_image = image.clone();
    int img_h = drawn_image.rows;
    int img_w = drawn_image.cols;

    for (const auto& det : detections) {
        cv::Scalar color = get_color(det.score);

        // Draw Bounding Box
        cv::rectangle(drawn_image,
                      cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                      color, 2);

        // Draw Label
        std::string label = "conf: " + cv::format("%.2f", det.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        int label_x = static_cast<int>(det.x1);
        int label_y = static_cast<int>(det.y1) - 5;
        if (label_y < text_size.height)
            label_y = static_cast<int>(det.y1) + text_size.height + 5;

        cv::rectangle(drawn_image,
                      cv::Point(label_x, label_y - text_size.height - baseline),
                      cv::Point(label_x + text_size.width, label_y + baseline),
                      color, cv::FILLED);

        int brightness = (color[0] + color[1] + color[2]) / 3;
        cv::Scalar text_color = brightness < 128 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);

        cv::putText(drawn_image, label,
                    cv::Point(label_x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv::LINE_AA);

        // Draw Keypoints
        for (int i = 0; i < 17; ++i) {
            if (det.kpt_conf[i] > 0.5f) {
                int kx = static_cast<int>(det.keypoints[i].first);
                int ky = static_cast<int>(det.keypoints[i].second);

                if (kx >= 0 && kx < img_w && ky >= 0 && ky < img_h) {
                    cv::circle(drawn_image, cv::Point(kx, ky), 4, cv::Scalar(0, 0, 255), -1);
                    cv::putText(drawn_image, KEYPOINT_NAMES[i],
                                cv::Point(kx + 5, ky - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
                }
            }
        }

        // Draw Skeleton
        for (const auto& pair : SKELETON) {
            int a = pair.first;
            int b = pair.second;

            if (det.kpt_conf[a] > 0.5f && det.kpt_conf[b] > 0.5f) {
                int x1 = static_cast<int>(det.keypoints[a].first);
                int y1 = static_cast<int>(det.keypoints[a].second);
                int x2 = static_cast<int>(det.keypoints[b].first);
                int y2 = static_cast<int>(det.keypoints[b].second);

                if (x1 >= 0 && x1 < img_w && y1 >= 0 && y1 < img_h &&
                    x2 >= 0 && x2 < img_w && y2 >= 0 && y2 < img_h) {
                    cv::line(drawn_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                }
            }
        }
    }
    return drawn_image;
}
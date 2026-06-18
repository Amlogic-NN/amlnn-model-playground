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
#include <numeric>


const int NUM_CLASSES = 13;
const std::string WORLD_CLASSES[13] = {
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", "skirt",
    "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
};

static float sigmoid(float x) {
    x = std::max(-250.0f, std::min(250.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

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

    return inter / (area1 + area2 - inter + 1e-6f);
}

static std::vector<Detection> nms_by_class(const std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return {};

    std::vector<Detection> final_detections;
    std::unordered_map<int, std::vector<Detection>> class_detections;

    for (const auto& det : detections) {
        class_detections[det.class_id].push_back(det);
    }

    for (auto& [class_id, cls_dets] : class_detections) {
        std::sort(cls_dets.begin(), cls_dets.end(), [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });

        std::vector<bool> removed(cls_dets.size(), false);
        for (size_t i = 0; i < cls_dets.size(); ++i) {
            if (removed[i]) continue;
            final_detections.push_back(cls_dets[i]);

            for (size_t j = i + 1; j < cls_dets.size(); ++j) {
                if (removed[j]) continue;
                if (compute_iou(cls_dets[i], cls_dets[j]) > iou_threshold) {
                    removed[j] = true;
                }
            }
        }
    }
    return final_detections;
}

static std::vector<Detection> suppress_cross_class_iou_conflicts(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return {};

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<bool> removed(detections.size(), false);
    std::vector<Detection> final_detections;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (removed[i]) continue;
        final_detections.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (removed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) {
                if (compute_iou(detections[i], detections[j]) > iou_threshold) {
                    removed[j] = true;
                }
            }
        }
    }
    return final_detections;
}

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape) {
    cv::Mat img_rgb;
    if (img.empty()) return {};

    if (img.channels() == 4) cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3) cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else img_rgb = img.clone();

    int orig_h = img.rows;
    int orig_w = img.cols;
    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);

    float scale = std::min(static_cast<float>(target_w) / orig_w,
                           static_cast<float>(target_h) / orig_h);

    int new_unpad_w = static_cast<int>(std::round(orig_w * scale));
    int new_unpad_h = static_cast<int>(std::round(orig_h * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

    float pad_w = (target_w - new_unpad_w) / 2.0f;
    float pad_h = (target_h - new_unpad_h) / 2.0f;

    int pad_left = static_cast<int>(std::round(pad_w - 0.1f));
    int pad_right = static_cast<int>(std::round(pad_w + 0.1f));
    int pad_top = static_cast<int>(std::round(pad_h - 0.1f));
    int pad_bottom = static_cast<int>(std::round(pad_h + 0.1f));

    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32FC3, 1.0 / 255.0);
    return std::make_tuple(img_float, scale, std::make_tuple(pad_left, pad_top));
}

cv::Mat quantize_input(const cv::Mat& float_img, float scale, int32_t zero_point, int tensor_type) {
    cv::Mat flat_img = float_img.isContinuous() ? float_img : float_img.clone();
    int total_elements = flat_img.total() * flat_img.channels();
    const float* src = (const float*)flat_img.data;

    cv::Mat quantized_img(1, total_elements, (tensor_type == 3) ? CV_8UC1 : CV_8SC1);

    if (tensor_type == 3) { // UInt8
        uint8_t* dst = (uint8_t*)quantized_img.data;
        for (int i = 0; i < total_elements; ++i) {
            dst[i] = static_cast<uint8_t>(std::clamp(std::nearbyint(src[i] / scale) + zero_point, 0.0f, 255.0f));
        }
    } else { // Int8
        int8_t* dst = (int8_t*)quantized_img.data;
        for (int i = 0; i < total_elements; ++i) {
            dst[i] = static_cast<int8_t>(std::clamp(std::nearbyint(src[i] / scale) + zero_point, -128.0f, 127.0f));
        }
    }

    return quantized_img;
}

std::vector<Detection> postprocess(
        const std::vector<float*>& out_buffers,
        const std::vector<std::vector<int>>& out_shapes,
        std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
        float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    int strides[3] = {8, 16, 32};
    std::vector<Detection> all_detections;
    float max_score_seen_overall = 0.0f;

    for (size_t out_idx = 0; out_idx < out_buffers.size(); ++out_idx) {
        float* data = out_buffers[out_idx];
        const auto& shape = out_shapes[out_idx];
        int stride = strides[out_idx];

        // STRICT: Only support exactly 77 channels
        int height = 0, width = 0;
        bool is_nchw = false;

        if (shape.size() >= 4 && shape[1] == 77) { // NCHW format
            is_nchw = true;
            height = shape[2];
            width = shape[3];
        } else if (shape.back() == 77) { // NHWC format
            is_nchw = false;
            height = shape[shape.size() - 3];
            width = shape[shape.size() - 2];
        } else {
            std::cerr << "[ERROR] Expected exactly 77 channels, but got unsupported shape layout!" << std::endl;
            continue; // Skip processing this layer
        }

        // Memory pointer helper hardcoded for 77 channels
        auto get_val = [&](int y, int x, int c) {
            if (is_nchw) return data[(c * height * width) + (y * width + x)];
            else         return data[(y * width + x) * 77 + c];
        };

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                float max_score = -1.0f;
                int class_id = -1;

                for (int c = 0; c < NUM_CLASSES; ++c) {
                    float val = get_val(y, x, 64 + c); // Classes are offset by 64
                    float score = sigmoid(val);
                    if (score > max_score) {
                        max_score = score;
                        class_id = c;
                    }
                }

                max_score_seen_overall = std::max(max_score_seen_overall, max_score);

                if (max_score >= conf_thresh) {
                    float bbox_deltas[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    for (int j = 0; j < 4; ++j) {
                        float max_dfl = -1e9f;
                        for (int k = 0; k < 16; ++k) {
                            max_dfl = std::max(max_dfl, get_val(y, x, j * 16 + k));
                        }

                        float expected_val = 0.0f;
                        float sum_exp = 0.0f;
                        for (int k = 0; k < 16; ++k) {
                            float exp_val = std::exp(get_val(y, x, j * 16 + k) - max_dfl);
                            sum_exp += exp_val;
                            expected_val += exp_val * k;
                        }
                        bbox_deltas[j] = expected_val / sum_exp;
                    }

                    float anchor_x = (x + 0.5f) * stride;
                    float anchor_y = (y + 0.5f) * stride;

                    float x1 = anchor_x - bbox_deltas[0] * stride;
                    float y1 = anchor_y - bbox_deltas[1] * stride;
                    float x2 = anchor_x + bbox_deltas[2] * stride;
                    float y2 = anchor_y + bbox_deltas[3] * stride;

                    float x1_orig = (x1 - pad_left) / scale;
                    float y1_orig = (y1 - pad_top) / scale;
                    float x2_orig = (x2 - pad_left) / scale;
                    float y2_orig = (y2 - pad_top) / scale;

                    all_detections.push_back({x1_orig, y1_orig, x2_orig, y2_orig, max_score, class_id});
                }
            }
        }
    }

    std::cout << "[DEBUG] Highest raw confidence score across entire image: " << max_score_seen_overall << std::endl;
    auto detections_nms = nms_by_class(all_detections, iou_threshold);
    return suppress_cross_class_iou_conflicts(detections_nms, 0.8f);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections) {
    cv::Mat drawn_image = image.clone();

    for (const auto& det : detections) {
        if (det.class_id < 0 || det.class_id >= NUM_CLASSES) continue;

        float hue = fmod(det.class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar color(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);

        cv::rectangle(drawn_image,
                      cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                      color, 2);

        std::string label = std::string(WORLD_CLASSES[det.class_id]) + ": " + cv::format("%.2f", det.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        int label_x = static_cast<int>(det.x1);
        int label_y = static_cast<int>(det.y1) - 5;
        if (label_y < text_size.height) label_y = static_cast<int>(det.y1) + text_size.height + 5;

        cv::rectangle(drawn_image,
                      cv::Point(label_x, label_y - text_size.height - baseline),
                      cv::Point(label_x + text_size.width, label_y + baseline),
                      color, cv::FILLED);

        int brightness = (color[0] + color[1] + color[2]) / 3;
        cv::Scalar text_color = brightness < 128 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);

        cv::putText(drawn_image, label, cv::Point(label_x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv::LINE_AA);
    }
    return drawn_image;
}
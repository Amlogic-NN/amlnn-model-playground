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
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>
#include <cstring>
#include <unordered_map>
#include "nn_sdk.h"

#define LOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)

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

static std::vector<Detection> suppress_cross_class_iou_conflicts(std::vector<Detection> detections, float iou_threshold) {
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
            if (detections[i].class_id != detections[j].class_id &&
                compute_iou(detections[i], detections[j]) > iou_threshold) {
                removed[j] = true;
            }
        }
    }
    return final_detections;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static std::vector<Detection> get_detections(float* output, std::tuple<int, int, int> output_shape, 
                                            int stride, float conf_thresh, int num_classes, int reverse) {
    std::vector<Detection> detections;

    int grid_h = std::get<0>(output_shape);
    int grid_w = std::get<1>(output_shape);
    int total_cells = grid_h * grid_w;
    int coords = 4 * 16;  // DFL coords: 64
    
    // reverse=0: standard YOLO [classes + box]
    // reverse>0: YOLOWorld [box + classes]
    int cls_offset = (reverse > 0) ? coords : 0;           
    int dfl_offset = (reverse > 0) ? 0 : num_classes;      

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            int idx = (i * grid_w + j) * (num_classes + coords);

            float max_score = -1.0f;
            int class_id = -1;
            for (int c = 0; c < num_classes; ++c) {
                int cls_idx = idx + cls_offset + c;
                float score = sigmoid(output[cls_idx]);
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score < conf_thresh) continue;

            float exp_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int k = 0; k < 4; ++k) {
                int dfl_idx = idx + dfl_offset + k * 16;
                float exp_logits[16];
                float sum_exp = 0.0f;

                float max_logit = output[dfl_idx];
                for (int t = 1; t < 16; ++t) {
                    if (output[dfl_idx + t] > max_logit) max_logit = output[dfl_idx + t];
                }

                for (int t = 0; t < 16; ++t) {
                    exp_logits[t] = std::exp(output[dfl_idx + t] - max_logit);
                    sum_exp += exp_logits[t];
                }

                for (int t = 0; t < 16; ++t) {
                    exp_logits[t] /= sum_exp;
                    exp_vals[k] += t * exp_logits[t];
                }
            }

            float x1 = (j + 0.5f - exp_vals[0]) * stride;
            float y1 = (i + 0.5f - exp_vals[1]) * stride;
            float x2 = (j + 0.5f + exp_vals[2]) * stride;
            float y2 = (i + 0.5f + exp_vals[3]) * stride;

            detections.push_back({x1, y1, x2, y2, max_score, class_id});
        }
    }
    return detections;
}


std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape) {
    cv::Mat img_rgb;
    // Check if image is valid
    if (img.empty()) {
        LOGE("Preprocess received empty image");
        return {};
    }

    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        img_rgb = img.clone(); 

    if (img.channels() == 3) {
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    }

    int orig_h = img.rows;
    int orig_w = img.cols;
    float scale = std::min(static_cast<float>(std::get<0>(new_shape)) / orig_h, static_cast<float>(std::get<1>(new_shape)) / orig_w);
    int new_h = static_cast<int>(round(orig_h * scale));
    int new_w = static_cast<int>(round(orig_w * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_h = std::get<0>(new_shape) - new_h;
    int pad_w = std::get<1>(new_shape) - new_w;
    int pad_left = static_cast<int>(round(pad_w / 2. - 0.1f));
    int pad_right = static_cast<int>(round(pad_w / 2. + 0.1f));
    int pad_top = static_cast<int>(round(pad_h / 2. - 0.1f));
    int pad_bottom = static_cast<int>(round(pad_h / 2. + 0.1f));

    cv::Mat img_padded;
    cv::copyMakeBorder(img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32F, 1.0 / 255.0);

    return std::make_tuple(img_float, scale, std::make_tuple(pad_left, pad_top));
}



std::vector<Detection> postprocess(std::tuple<float*, std::tuple<int, int, int>, int> out0,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out1,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out2,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold, int num_classes, int reverse) {
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    std::vector<Detection> detections;

    auto process_out = [&](auto& out) {
        float* output = std::get<0>(out);
        auto shape = std::get<1>(out);
        int stride = std::get<2>(out);
        std::vector<Detection> dets = get_detections(output, shape, stride, conf_thresh, num_classes, reverse);
        detections.insert(detections.end(), dets.begin(), dets.end());
    };

    process_out(out0);
    process_out(out1);
    process_out(out2);

    std::vector<Detection> detections_orig;
    for (const auto& det : detections) {
        float x1_orig = (det.x1 - pad_left) / scale;
        float y1_orig = (det.y1 - pad_top) / scale;
        float x2_orig = (det.x2 - pad_left) / scale;
        float y2_orig = (det.y2 - pad_top) / scale;
        detections_orig.push_back({x1_orig, y1_orig, x2_orig, y2_orig, det.score, det.class_id});
    }

    std::vector<Detection> detections_nms = nms_by_class(detections_orig, iou_threshold);
    return suppress_cross_class_iou_conflicts(detections_nms, 0.8f);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections, 
                        const std::vector<std::string>& classes, int seed_offset) {
    int num_classes = classes.size();
    std::vector<cv::Scalar> color_palette;
    std::mt19937 rng(42 + seed_offset);
    std::uniform_int_distribution<int> color_dist(0, 255);

    for (int i = 0; i < num_classes; ++i) {
        color_palette.emplace_back(color_dist(rng), color_dist(rng), color_dist(rng));
    }

    cv::Mat drawn_image = image.clone();

    for (const auto& det : detections) {
        int class_id = det.class_id;
        if (class_id < 0 || class_id >= num_classes) continue;

        cv::Scalar color = color_palette[class_id];
        cv::rectangle(drawn_image, 
                      cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)), 
                      color, 2);

        std::string label = classes[class_id] + ": " + cv::format("%.2f", det.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int label_x = static_cast<int>(det.x1);
        int label_y = static_cast<int>(det.y1) - 10;
        if (label_y < text_size.height) label_y = static_cast<int>(det.y1) + text_size.height + 10;

        cv::rectangle(drawn_image, 
                      cv::Point(label_x, label_y - text_size.height - baseline),
                      cv::Point(label_x + text_size.width, label_y + baseline), 
                      color, cv::FILLED);

        cv::putText(drawn_image, label, 
                    cv::Point(label_x, label_y), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    return drawn_image;
}

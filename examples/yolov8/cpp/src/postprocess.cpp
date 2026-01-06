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

#define LOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)

// COCO class names (80 classes)
const char* COCO_CLASSES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape) {
    cv::Mat img_rgb;
    
    if (img.empty()) {
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
    cv::copyMakeBorder(img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right, 
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32F, 1.0 / 255.0);

    return std::make_tuple(img_float, scale, std::make_tuple(pad_left, pad_top));
}

cv::Mat quantize_input(const cv::Mat& float_img, float scale, int8_t zero_point) {
    if (float_img.empty() || float_img.type() != CV_32FC3) {
        LOGE("quantize_input: Invalid input image (must be CV_32FC3)");
        return cv::Mat();
    }

    cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8SC3);
    const float* src_ptr = (const float*)float_img.data;
    int8_t* dst_ptr = (int8_t*)quantized_img.data;

    int total_elements = float_img.total() * float_img.channels();
    for (int i = 0; i < total_elements; ++i) {
        dst_ptr[i] = static_cast<int8_t>(std::round(src_ptr[i] / scale + zero_point));
    }

    return quantized_img;
}

static std::vector<Detection> get_detections(float* output, std::tuple<int, int, int> output_shape, 
                                            int stride, float conf_thresh) {
    std::vector<Detection> detections;

    int grid_h = std::get<0>(output_shape);
    int grid_w = std::get<1>(output_shape);
    int channels = std::get<2>(output_shape);
    
    const int num_classes = 80;
    const int dfl_channels = 64;  // 4 directions * 16 bins

    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            // NHWC format: output[i][j][c]
            int base_idx = (i * grid_w + j) * channels;

            float max_score = -1.0f;
            int class_id = -1;
            for (int c = 0; c < num_classes; ++c) {
                float score = sigmoid(output[base_idx + dfl_channels + c]);
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score < conf_thresh) continue;

            // DFL decoding for bounding box
            float bbox_deltas[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int k = 0; k < 4; ++k) {
                int dfl_start = base_idx + k * 16;

                // Softmax over 16 bins
                float exp_logits[16];
                float max_logit = output[dfl_start];
                for (int t = 1; t < 16; ++t) {
                    if (output[dfl_start + t] > max_logit)
                        max_logit = output[dfl_start + t];
                }

                float sum_exp = 0.0f;
                for (int t = 0; t < 16; ++t) {
                    exp_logits[t] = std::exp(output[dfl_start + t] - max_logit);
                    sum_exp += exp_logits[t];
                }

                for (int t = 0; t < 16; ++t) {
                    exp_logits[t] /= sum_exp;
                    bbox_deltas[k] += t * exp_logits[t];
                }
            }

            // Convert to absolute coordinates
            float anchor_x = (j + 0.5f) * stride;
            float anchor_y = (i + 0.5f) * stride;

            float left = bbox_deltas[0];
            float top = bbox_deltas[1];
            float right = bbox_deltas[2];
            float bottom = bbox_deltas[3];

            float x1 = anchor_x - left * stride;
            float y1 = anchor_y - top * stride;
            float x2 = anchor_x + right * stride;
            float y2 = anchor_y + bottom * stride;

            detections.push_back({x1, y1, x2, y2, max_score, class_id});
        }
    }
    return detections;
}

std::vector<Detection> postprocess(std::tuple<float*, std::tuple<int, int, int>, int> out0,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out1,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out2,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold) {
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    std::vector<Detection> detections;

    auto process_out = [&](auto& out) {
        float* output = std::get<0>(out);
        auto shape = std::get<1>(out);
        int stride = std::get<2>(out);
        std::vector<Detection> dets = get_detections(output, shape, stride, conf_thresh);
        detections.insert(detections.end(), dets.begin(), dets.end());
    };

    // Process all three scales
    process_out(out0);
    process_out(out1);
    process_out(out2);

    // Map coordinates back to original image
    std::vector<Detection> detections_orig;
    for (const auto& det : detections) {
        float x1_orig = (det.x1 - pad_left) / scale;
        float y1_orig = (det.y1 - pad_top) / scale;
        float x2_orig = (det.x2 - pad_left) / scale;
        float y2_orig = (det.y2 - pad_top) / scale;

        // Clamp to non-negative
        x1_orig = std::max(0.0f, x1_orig);
        y1_orig = std::max(0.0f, y1_orig);
        x2_orig = std::max(0.0f, x2_orig);
        y2_orig = std::max(0.0f, y2_orig);

        detections_orig.push_back({x1_orig, y1_orig, x2_orig, y2_orig, det.score, det.class_id});
    }

    // Apply NMS
    return nms_by_class(detections_orig, iou_threshold);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections) {
    cv::Mat drawn_image = image.clone();

    for (const auto& det : detections) {
        int class_id = det.class_id;
        if (class_id < 0 || class_id >= 80) continue;

        // Generate color based on class_id using HSV
        float hue = fmod(class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar color(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);

        // Draw bounding box
        cv::rectangle(drawn_image, 
                      cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)), 
                      color, 2);

        // Draw label
        std::string label = std::string(COCO_CLASSES[class_id]) + ": " + cv::format("%.2f", det.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        int label_x = static_cast<int>(det.x1);
        int label_y = static_cast<int>(det.y1) - 5;
        if (label_y < text_size.height)
            label_y = static_cast<int>(det.y1) + text_size.height + 5;

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

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

#define LOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)

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
    "broccoli", "carrot", "hot dog", "pizza", "doughnut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
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

std::vector<Detection> postprocess(float* bbox_data, const std::vector<int>& bbox_shape,
                                   float* score_data, const std::vector<int>& score_shape,
                                   float* mask_coeff_data, const std::vector<int>& mask_coeff_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    std::vector<Detection> detections_orig;

    // Detect layouts dynamically
    bool bbox_channels_last = (bbox_shape.size() > 1 && bbox_shape[1] == 4);
    int num_anchors = bbox_channels_last ? bbox_shape[0] : (bbox_shape.size() > 1 ? bbox_shape[1] : 8400);

    bool score_channels_last = (score_shape.size() > 1 && score_shape[1] == 80);
    bool mask_channels_last = (mask_coeff_shape.size() > 1 && mask_coeff_shape[1] == 32);

    for (int i = 0; i < num_anchors; ++i) {
        float max_score = 0.0f;
        int class_id = -1;

        // 1. Find Highest Score
        for (int c = 0; c < 80; ++c) {
            int idx = score_channels_last ? (i * 80 + c) : (c * num_anchors + i);
            float score = score_data[idx];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        // 2. Process Bbox & Mask if confident
        if (max_score >= conf_thresh) {
            int idx_cx = bbox_channels_last ? (i * 4 + 0) : (0 * num_anchors + i);
            int idx_cy = bbox_channels_last ? (i * 4 + 1) : (1 * num_anchors + i);
            int idx_w  = bbox_channels_last ? (i * 4 + 2) : (2 * num_anchors + i);
            int idx_h  = bbox_channels_last ? (i * 4 + 3) : (3 * num_anchors + i);

            float cx = bbox_data[idx_cx];
            float cy = bbox_data[idx_cy];
            float w  = bbox_data[idx_w];
            float h  = bbox_data[idx_h];

            float x1 = cx - (w / 2.0f);
            float y1 = cy - (h / 2.0f);
            float x2 = cx + (w / 2.0f);
            float y2 = cy + (h / 2.0f);

            float x1_orig = std::max(0.0f, (x1 - pad_left) / scale);
            float y1_orig = std::max(0.0f, (y1 - pad_top) / scale);
            float x2_orig = std::max(0.0f, (x2 - pad_left) / scale);
            float y2_orig = std::max(0.0f, (y2 - pad_top) / scale);

            // Extract the 32 mask coefficients
            std::vector<float> coeffs(32);
            for (int m = 0; m < 32; ++m) {
                int idx = mask_channels_last ? (i * 32 + m) : (m * num_anchors + i);
                coeffs[m] = mask_coeff_data[idx];
            }

            detections_orig.push_back({x1_orig, y1_orig, x2_orig, y2_orig, max_score, class_id, coeffs});
        }
    }

    return nms_by_class(detections_orig, iou_threshold);
}


cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections,
                        float* proto_mask_data, const std::vector<int>& proto_shape,
                        float scale, std::tuple<int, int> pad){
    cv::Mat drawn_image = image.clone();

    int pad_left = std::get<0>(pad);
    int pad_top = std::get<1>(pad);
    int input_size = 640;

    // Figure out Proto Mask layout. Python usually outputs [160, 160, 32] (NHWC)
    // or [32, 160, 160] (NCHW).
    bool proto_channels_last = (proto_shape.size() == 3 && proto_shape[2] == 32);

    cv::Mat proto_mat;
    if (proto_channels_last) {
        // [160*160, 32] layout
        proto_mat = cv::Mat(160 * 160, 32, CV_32F, proto_mask_data);
    } else {
        // [32, 160*160] layout
        proto_mat = cv::Mat(32, 160 * 160, CV_32F, proto_mask_data);
    }

    for (const auto& det : detections) {
        int class_id = det.class_id;
        if (class_id < 0 || class_id >= 80) continue;

        // Generate color based on class_id
        float hue = fmod(class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar color(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);

        // === 1. PROCESS MASKS ===
        cv::Mat coeff_mat(1, 32, CV_32F, (void*)det.mask_coeff.data());
        cv::Mat raw_mask;

        // Matrix multiplication (dot product of mask_coeff and proto_mask)
        if (proto_channels_last) {
            // (1 x 32) * (32 x 25600) -> (1 x 25600)
            raw_mask = coeff_mat * proto_mat.t();
        } else {
            // (1 x 32) * (32 x 25600) -> (1 x 25600)
            raw_mask = coeff_mat * proto_mat;
        }

        // Reshape to 160x160 and apply Sigmoid
        raw_mask = raw_mask.reshape(1, 160);
        cv::Mat exp_mat;
        cv::exp(-raw_mask, exp_mat);
        cv::Mat mask_sigmoid = 1.0 / (1.0 + exp_mat);

        // Upscale to model input size (640x640)
        cv::Mat mask_640;
        cv::resize(mask_sigmoid, mask_640, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);

        // Crop Letterbox padding
        int real_w = std::round(image.cols * scale);
        int real_h = std::round(image.rows * scale);

        cv::Rect roi(pad_left, pad_top, real_w, real_h);
        // Safety bound constraints
        roi.x = std::max(0, roi.x);
        roi.y = std::max(0, roi.y);
        roi.width = std::min(input_size - roi.x, roi.width);
        roi.height = std::min(input_size - roi.y, roi.height);

        cv::Mat cropped_mask = mask_640(roi);

        // Scale back to original image dimensions
        cv::Mat final_mask;
        cv::resize(cropped_mask, final_mask, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);

        // Generate Boolean Binary Mask (> 0.5)
        cv::Mat binary_mask = final_mask > 0.5f;

        // Apply Green overlay to original image via Alpha Blend
        cv::Mat colored_mask = cv::Mat::zeros(image.size(), image.type());
        colored_mask.setTo(color, binary_mask);

        cv::Mat blended;
        cv::addWeighted(drawn_image, 1.0, colored_mask, 0.5, 0, blended); // 0.5 Alpha

        // Copy blended pixels only where mask exists to keep the rest intact
        blended.copyTo(drawn_image, binary_mask);

        // === 2. DRAW BOUNDING BOX & LABEL ===
        cv::rectangle(drawn_image,
                      cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                      color, 2);

        std::string label = std::string(COCO_CLASSES[class_id]) + ": " + cv::format("%.2f", det.score);
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
    }
    return drawn_image;
}
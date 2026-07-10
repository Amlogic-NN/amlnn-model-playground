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

const std::vector<std::string> DOTA_CLASSES = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool"
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
    float xx1 = std::max(det1.aabb_x1, det2.aabb_x1);
    float yy1 = std::max(det1.aabb_y1, det2.aabb_y1);
    float xx2 = std::min(det1.aabb_x2, det2.aabb_x2);
    float yy2 = std::min(det1.aabb_y2, det2.aabb_y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area1 = (det1.aabb_x2 - det1.aabb_x1) * (det1.aabb_y2 - det1.aabb_y1);
    float area2 = (det2.aabb_x2 - det2.aabb_x1) * (det2.aabb_y2 - det2.aabb_y1);

    return inter / (area1 + area2 - inter);
}

static std::vector<Detection> nms_by_class(std::vector<Detection>& detections, float iou_threshold) {
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

// Memory parsing tool robust against model layouts (NHWC vs NCHW)
std::vector<int> get_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

int get_idx_from_shape(const std::vector<int>& shape, const std::vector<int>& strides, int val_c, int val_n) {
    int offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 1000) offset += val_n * strides[i]; // N anchors dimension (21504)
        else if (shape[i] > 1) offset += val_c * strides[i]; // Class/Bbox channel dimension
    }
    return offset;
}

std::vector<Detection> postprocess(float* bbox_data, const std::vector<int>& bbox_shape,
                                   float* score_data, const std::vector<int>& score_shape,
                                   float* angle_data, const std::vector<int>& angle_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    int num_anchors = 21504; // Standard for 1024x1024 input
    for (int d : score_shape) { if (d > 1000) { num_anchors = d; break; } }

    std::vector<int> stride_bbox = get_strides(bbox_shape);
    std::vector<int> stride_score = get_strides(score_shape);
    std::vector<int> stride_angle = get_strides(angle_shape);

    std::vector<Detection> detections_orig;

    for (int i = 0; i < num_anchors; ++i) {
        float max_score = 0.0f;
        int class_id = -1;

        // 1. Find the highest class score
        for (int c = 0; c < 15; ++c) {
            float score = score_data[get_idx_from_shape(score_shape, stride_score, c, i)];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= conf_thresh) {
            float cx = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 0, i)];
            float cy = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 1, i)];
            float w  = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 2, i)];
            float h  = bbox_data[get_idx_from_shape(bbox_shape, stride_bbox, 3, i)];
            float angle_rad = angle_data[get_idx_from_shape(angle_shape, stride_angle, 0, i)];

            // Undo Letterbox Padding & Scale
            cx = (cx - pad_left) / scale;
            cy = (cy - pad_top) / scale;
            w /= scale;
            h /= scale;

            // Convert to degrees
            float angle_deg = angle_rad * (180.0f / CV_PI);

            // Calculate Oriented Box Corners using OpenCV
            cv::RotatedRect rect(cv::Point2f(cx, cy), cv::Size2f(w, h), angle_deg);
            cv::Point2f pts[4];
            rect.points(pts);

            // Compute Axis Aligned Box for NMS
            float min_x = pts[0].x, min_y = pts[0].y;
            float max_x = pts[0].x, max_y = pts[0].y;
            for (int k = 1; k < 4; ++k) {
                min_x = std::min(min_x, pts[k].x);
                min_y = std::min(min_y, pts[k].y);
                max_x = std::max(max_x, pts[k].x);
                max_y = std::max(max_y, pts[k].y);
            }

            Detection det;
            det.corners = {pts[0], pts[1], pts[2], pts[3]};
            det.aabb_x1 = min_x;
            det.aabb_y1 = min_y;
            det.aabb_x2 = max_x;
            det.aabb_y2 = max_y;
            det.score = max_score;
            det.class_id = class_id;

            detections_orig.push_back(det);
        }
    }

    return nms_by_class(detections_orig, iou_threshold);
}

cv::Scalar get_color(int class_id) {
    float hue = fmod(class_id * 137.508f, 360.0f);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections) {
    cv::Mat drawn_image = image.clone();

    for (const auto& det : detections) {
        int class_id = det.class_id;
        cv::Scalar color = get_color(class_id);

        // Map float corners to Integer points for drawing
        std::vector<cv::Point> int_corners(4);
        int top_idx = 0;
        float min_y = det.corners[0].y;

        for (int i = 0; i < 4; ++i) {
            int_corners[i] = cv::Point(static_cast<int>(std::round(det.corners[i].x)),
                                       static_cast<int>(std::round(det.corners[i].y)));

            // Find the highest point on screen to place the text label nicely
            if (det.corners[i].y < min_y) {
                min_y = det.corners[i].y;
                top_idx = i;
            }
        }

        // Draw Oriented Bounding Box
        std::vector<std::vector<cv::Point>> polys = {int_corners};
        cv::polylines(drawn_image, polys, true, color, 2);

        // Draw Label Text
        std::string class_name = (class_id >= 0 && class_id < DOTA_CLASSES.size()) ? DOTA_CLASSES[class_id] : "Unknown";
        std::string label = class_name + ": " + cv::format("%.2f", det.score);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        int label_x = int_corners[top_idx].x;
        int label_y = int_corners[top_idx].y - 5;

        // Draw background rectangle for text
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
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

const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "doughnut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

const float ANCHORS[3][3][2] = {
    {{10, 13}, {16, 30}, {33, 23}},      // Stride 8
    {{30, 61}, {62, 45}, {59, 119}},     // Stride 16
    {{116, 90}, {156, 198}, {373, 326}}  // Stride 32
};

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

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<Detection> postprocess(float* out0_data, const std::vector<int>& out0_shape,
                                   float* out1_data, const std::vector<int>& out1_shape,
                                   float* out2_data, const std::vector<int>& out2_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    std::vector<Detection> detections_orig;

    auto process_branch = [&](float* data, const std::vector<int>& shape) {
        int total_elements = 1;
        for (int d : shape) total_elements *= d;

        // Dynamically deduce grid size from total memory block assuming 255 channels
        int grid_size = static_cast<int>(std::round(std::sqrt(total_elements / 255.0f)));
        int stride = 640 / grid_size;
        int anchor_idx = (stride == 8) ? 0 : (stride == 16) ? 1 : 2;

        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                for (int a = 0; a < 3; ++a) {
                    // Standard NHWC memory layout index
                    int base_idx = y * (grid_size * 255) + x * 255 + a * 85;

                    float obj_conf = sigmoid(data[base_idx + 4]);
                    if (obj_conf < conf_thresh) continue;

                    float max_cls_prob = 0.0f;
                    int best_cls = -1;

                    for (int c = 0; c < 80; ++c) {
                        float cls_prob = sigmoid(data[base_idx + 5 + c]);
                        if (cls_prob > max_cls_prob) {
                            max_cls_prob = cls_prob;
                            best_cls = c;
                        }
                    }

                    float final_score = obj_conf * max_cls_prob;
                    if (final_score >= conf_thresh) {
                        float tx = data[base_idx + 0];
                        float ty = data[base_idx + 1];
                        float tw = data[base_idx + 2];
                        float th = data[base_idx + 3];

                        float bx = (sigmoid(tx) * 2.0f - 0.5f + x) * stride;
                        float by = (sigmoid(ty) * 2.0f - 0.5f + y) * stride;
                        float bw = pow(sigmoid(tw) * 2.0f, 2) * ANCHORS[anchor_idx][a][0];
                        float bh = pow(sigmoid(th) * 2.0f, 2) * ANCHORS[anchor_idx][a][1];

                        float x1 = (bx - bw / 2.0f - pad_left) / scale;
                        float y1 = (by - bh / 2.0f - pad_top) / scale;
                        float x2 = (bx + bw / 2.0f - pad_left) / scale;
                        float y2 = (by + bh / 2.0f - pad_top) / scale;

                        Detection det;
                        det.x1 = std::max(0.0f, x1);
                        det.y1 = std::max(0.0f, y1);
                        det.x2 = std::max(0.0f, x2);
                        det.y2 = std::max(0.0f, y2);
                        det.score = final_score;
                        det.class_id = best_cls;
                        det.class_name = COCO_CLASSES[best_cls];

                        detections_orig.push_back(det);
                    }
                }
            }
        }
    };

    process_branch(out0_data, out0_shape);
    process_branch(out1_data, out1_shape);
    process_branch(out2_data, out2_shape);

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

    for (const auto& det : detections) {
        cv::Scalar color = get_color(det.score);

        cv::rectangle(drawn_image,
                      cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)),
                      color, 2);

        std::string label = det.class_name + " " + cv::format("%.2f", det.score);
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
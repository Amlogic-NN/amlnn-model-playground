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

#define LOGI(...)            \
    do                       \
    {                        \
        printf(__VA_ARGS__); \
        printf("\n");        \
    } while (0)
#define LOGE(...)                     \
    do                                \
    {                                 \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)

const char *COCO_CLASSES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "doughnut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

// ImageNet Mean/Std values
const float MEAN[3] = {123.675f, 116.28f, 103.53f};
const float STD[3] = {58.395f, 57.12f, 57.37f};

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

static float compute_iou(const Detection &det1, const Detection &det2)
{
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

static std::vector<Detection> nms_by_class(const std::vector<Detection> &detections, float iou_threshold)
{
    if (detections.empty())
        return {};

    std::vector<Detection> final_detections;
    std::unordered_map<int, std::vector<Detection>> class_detections;
    for (const auto &det : detections)
    {
        class_detections[det.class_id].push_back(det);
    }

    for (auto &[class_id, cls_dets] : class_detections)
    {
        std::sort(cls_dets.begin(), cls_dets.end(), [](const Detection &a, const Detection &b)
                  { return a.score > b.score; });

        std::vector<bool> removed(cls_dets.size(), false);
        for (size_t i = 0; i < cls_dets.size(); ++i)
        {
            if (removed[i])
                continue;
            final_detections.push_back(cls_dets[i]);

            for (size_t j = i + 1; j < cls_dets.size(); ++j)
            {
                if (removed[j])
                    continue;
                if (compute_iou(cls_dets[i], cls_dets[j]) > iou_threshold)
                {
                    removed[j] = true;
                }
            }
        }
    }
    return final_detections;
}

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape)
{
    cv::Mat img_rgb;

    if (img.empty())
    {
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

    cv::Mat img_float(img_padded.size(), CV_32FC3);

    // Normalization
    for (int y = 0; y < img_padded.rows; ++y)
    {
        for (int x = 0; x < img_padded.cols; ++x)
        {
            cv::Vec3b pixel = img_padded.at<cv::Vec3b>(y, x);
            img_float.at<cv::Vec3f>(y, x) = cv::Vec3f(
                (pixel[0] - MEAN[0]) / STD[0],
                (pixel[1] - MEAN[1]) / STD[1],
                (pixel[2] - MEAN[2]) / STD[2]);
        }
    }

    return std::make_tuple(img_float, scale, std::make_tuple(pad_left, pad_top));
}

cv::Mat quantize_input(const cv::Mat &float_img, float scale, int32_t zero_point)
{
    if (float_img.empty() || float_img.type() != CV_32FC3)
        return cv::Mat();

    cv::Mat quantized_img(float_img.rows, float_img.cols, CV_8SC3);
    const float *src_ptr = (const float *)float_img.data;
    int8_t *dst_ptr = (int8_t *)quantized_img.data;

    int total_elements = float_img.total() * float_img.channels();
    for (int i = 0; i < total_elements; ++i)
    {
        // Quantization
        float val = std::round(src_ptr[i] / scale) + zero_point;
        dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
    }
    return quantized_img;
}

std::vector<Detection> postprocess(float *output_data, const std::vector<int> &output_shape,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    // Dynamic layout check: Is it [8400, 85] or [85, 8400]
    bool channels_last = (output_shape.back() == 85);
    int num_anchors = channels_last ? output_shape[output_shape.size() - 2] : output_shape.back();

    std::vector<Detection> detections_orig;

    for (int i = 0; i < num_anchors; ++i)
    {
        // Read object confidence
        int obj_idx = channels_last ? (i * 85 + 4) : (4 * num_anchors + i);
        float obj_conf = output_data[obj_idx];

        // Early stopping
        if (obj_conf < conf_thresh)
            continue;

        // Find Class ID with highest score
        float max_cls_score = 0.0f;
        int class_id = -1;
        for (int c = 0; c < 80; ++c)
        {
            int cls_idx = channels_last ? (i * 85 + 5 + c) : ((5 + c) * num_anchors + i);
            float cls_score = output_data[cls_idx];
            if (cls_score > max_cls_score)
            {
                max_cls_score = cls_score;
                class_id = c;
            }
        }

        // Final score
        float final_score = obj_conf * max_cls_score;

        if (final_score >= conf_thresh)
        {
            // 1. Calculate Grid X, Y, and Stride
            int grid_x, grid_y, stride;
            if (i < 6400)
            {
                stride = 8;
                grid_y = i / 80;
                grid_x = i % 80;
            }
            else if (i < 8000)
            {
                stride = 16;
                int local_i = i - 6400;
                grid_y = local_i / 40;
                grid_x = local_i % 40;
            }
            else
            {
                stride = 32;
                int local_i = i - 8000;
                grid_y = local_i / 20;
                grid_x = local_i % 20;
            }

            int idx_cx = channels_last ? (i * 85 + 0) : (0 * num_anchors + i);
            int idx_cy = channels_last ? (i * 85 + 1) : (1 * num_anchors + i);
            int idx_w = channels_last ? (i * 85 + 2) : (2 * num_anchors + i);
            int idx_h = channels_last ? (i * 85 + 3) : (3 * num_anchors + i);

            float raw_cx = output_data[idx_cx];
            float raw_cy = output_data[idx_cy];
            float raw_w = output_data[idx_w];
            float raw_h = output_data[idx_h];

            // 2. YOLOX Grid Decoding using our calculated variables
            float cx = (raw_cx + grid_x) * stride;
            float cy = (raw_cy + grid_y) * stride;
            float w = std::exp(raw_w) * stride;
            float h = std::exp(raw_h) * stride;

            float x1 = cx - (w / 2.0f);
            float y1 = cy - (h / 2.0f);
            float x2 = cx + (w / 2.0f);
            float y2 = cy + (h / 2.0f);

            // Reverse Letterbox Pad & Scale
            float x1_orig = std::max(0.0f, (x1 - pad_left) / scale);
            float y1_orig = std::max(0.0f, (y1 - pad_top) / scale);
            float x2_orig = std::max(0.0f, (x2 - pad_left) / scale);
            float y2_orig = std::max(0.0f, (y2 - pad_top) / scale);

            detections_orig.push_back({x1_orig, y1_orig, x2_orig, y2_orig, final_score, class_id});
        }
    }

    return nms_by_class(detections_orig, iou_threshold);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection> &detections)
{
    cv::Mat drawn_image = image.clone();

    for (const auto &det : detections)
    {
        int class_id = det.class_id;
        if (class_id < 0 || class_id >= 80)
            continue;

        // Generate color based on class_id using HSV
        float hue = fmod(class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar color(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);

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
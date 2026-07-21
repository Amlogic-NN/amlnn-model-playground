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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>

const char *COCO_CLASSES[80] = {
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
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

// Output order is stride 8, stride 16, then stride 32.
const int STRIDES[3] = {8, 16, 32};
const float ANCHORS[3][3][2] = {
    {{12, 16}, {19, 36}, {40, 28}},
    {{36, 75}, {76, 55}, {72, 146}},
    {{142, 110}, {192, 243}, {459, 401}}};

static float sigmoid(float value)
{
    value = std::max(-80.0f, std::min(80.0f, value));
    return 1.0f / (1.0f + std::exp(-value));
}

static float compute_iou(const Detection &det1, const Detection &det2)
{
    float xx1 = std::max(det1.x1, det2.x1);
    float yy1 = std::max(det1.y1, det2.y1);
    float xx2 = std::min(det1.x2, det2.x2);
    float yy2 = std::min(det1.y2, det2.y2);
    float width = std::max(0.0f, xx2 - xx1);
    float height = std::max(0.0f, yy2 - yy1);
    float intersection = width * height;
    float area1 = std::max(0.0f, det1.x2 - det1.x1) * std::max(0.0f, det1.y2 - det1.y1);
    float area2 = std::max(0.0f, det2.x2 - det2.x1) * std::max(0.0f, det2.y2 - det2.y1);
    float union_area = area1 + area2 - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static std::vector<Detection> nms_by_class(
    const std::vector<Detection> &detections, float iou_threshold)
{
    if (detections.empty())
        return {};

    std::vector<Detection> final_detections;
    std::unordered_map<int, std::vector<Detection>> class_detections;
    for (const auto &det : detections)
        class_detections[det.class_id].push_back(det);

    for (auto &[class_id, class_dets] : class_detections)
    {
        std::sort(
            class_dets.begin(), class_dets.end(),
            [](const Detection &a, const Detection &b)
            { return a.score > b.score; });
        std::vector<bool> removed(class_dets.size(), false);

        for (size_t i = 0; i < class_dets.size(); ++i)
        {
            if (removed[i])
                continue;

            final_detections.push_back(class_dets[i]);
            for (size_t j = i + 1; j < class_dets.size(); ++j)
            {
                if (!removed[j] && compute_iou(class_dets[i], class_dets[j]) > iou_threshold)
                    removed[j] = true;
            }
        }
    }

    return final_detections;
}

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
            shape.push_back(attr.dims[i]);
    }
    return shape;
}

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(
    cv::Mat img, std::tuple<int, int> new_shape)
{
    if (img.empty())
        return {};

    cv::Mat img_rgb;
    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int original_h = img.rows;
    int original_w = img.cols;
    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);
    float scale = std::min(
        static_cast<float>(target_h) / original_h,
        static_cast<float>(target_w) / original_w);
    int resized_h = static_cast<int>(std::round(original_h * scale));
    int resized_w = static_cast<int>(std::round(original_w * scale));

    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    int pad_h = target_h - resized_h;
    int pad_w = target_w - resized_w;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;

    cv::Mat img_padded;
    cv::copyMakeBorder(
        img_resized, img_padded, pad_top, pad_bottom, pad_left, pad_right,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat img_float;
    img_padded.convertTo(img_float, CV_32F, 1.0 / 255.0);
    return std::make_tuple(img_float, scale, std::make_tuple(pad_left, pad_top));
}

std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;
    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        std::cerr << "prepare_input_tensor: Invalid input image" << std::endl;
        return tensor_data;
    }

    int total_elements = static_cast<int>(float_img.total() * float_img.channels());
    const float *src_ptr = float_img.ptr<float>();

    // Quantize with the tensor scale/zero point and saturate to the target type.
    if (attr.type == AMLNN_TENSOR_FLOAT32)
    {
        tensor_data.resize(total_elements * sizeof(float));
        std::memcpy(tensor_data.data(), float_img.data, tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(total_elements * sizeof(int16_t));
        int16_t *dst_ptr = reinterpret_cast<int16_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, value)));
        }
    }
    else
    {
        std::cerr << "prepare_input_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

std::vector<Detection> postprocess(
    const std::vector<float *> &out_ptrs, const std::vector<std::vector<int>> &out_shapes,
    int input_h, int input_w, std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
    float conf_thresh, float iou_threshold)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));
    std::vector<Detection> detections_orig;

    float safe_thresh = std::max(1e-5f, std::min(conf_thresh, 1.0f - 1e-5f));
    float inv_thresh = std::log(safe_thresh / (1.0f - safe_thresh));

    for (int s = 0; s < 3; ++s)
    {
        float *data = out_ptrs[s];
        const auto &shape = out_shapes[s];
        int stride = STRIDES[s];
        int height = 1;
        int width = 1;
        int channels = 1;

        if (shape.size() == 4)
        {
            height = shape[1];
            width = shape[2];
            channels = shape[3];
        }
        else if (shape.size() == 3)
        {
            height = shape[0];
            width = shape[1];
            channels = shape[2];
        }
        else
        {
            std::cerr << "Unexpected output shape for output " << s << std::endl;
            continue;
        }

        const int num_anchors = 3;
        if (channels % num_anchors != 0)
        {
            std::cerr << "Invalid YOLOv7 channel count: " << channels << std::endl;
            continue;
        }

        int values_per_anchor = channels / num_anchors;
        int num_classes = values_per_anchor - 5;
        if (num_classes <= 0)
        {
            std::cerr << "Invalid values per anchor: " << values_per_anchor << std::endl;
            continue;
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const float *cell_data = data + (y * width + x) * channels;

                for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx)
                {
                    const float *anchor_data = cell_data + anchor_idx * values_per_anchor;
                    float raw_objectness = anchor_data[4];
                    if (raw_objectness <= inv_thresh)
                        continue;

                    float max_raw_class = -1e9f;
                    int class_id = -1;
                    for (int class_idx = 0; class_idx < num_classes; ++class_idx)
                    {
                        float value = anchor_data[5 + class_idx];
                        if (value > max_raw_class)
                        {
                            max_raw_class = value;
                            class_id = class_idx;
                        }
                    }

                    float final_score = sigmoid(raw_objectness) * sigmoid(max_raw_class);
                    if (final_score <= conf_thresh)
                        continue;

                    // Standard YOLOv7 anchor decode.
                    float tx = sigmoid(anchor_data[0]);
                    float ty = sigmoid(anchor_data[1]);
                    float tw = sigmoid(anchor_data[2]);
                    float th = sigmoid(anchor_data[3]);
                    float center_x = (tx * 2.0f - 0.5f + x) * stride;
                    float center_y = (ty * 2.0f - 0.5f + y) * stride;
                    float box_w = tw * 2.0f;
                    float box_h = th * 2.0f;
                    box_w = box_w * box_w * ANCHORS[s][anchor_idx][0];
                    box_h = box_h * box_h * ANCHORS[s][anchor_idx][1];

                    float x1 = (center_x - box_w * 0.5f - pad_left) / scale;
                    float y1 = (center_y - box_h * 0.5f - pad_top) / scale;
                    float x2 = (center_x + box_w * 0.5f - pad_left) / scale;
                    float y2 = (center_y + box_h * 0.5f - pad_top) / scale;
                    detections_orig.push_back({std::max(0.0f, x1), std::max(0.0f, y1),
                                               std::max(0.0f, x2), std::max(0.0f, y2),
                                               final_score, class_id});
                }
            }
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

        float hue = std::fmod(class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat rgb;
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
        cv::Scalar color(
            rgb.at<cv::Vec3b>(0, 0)[0],
            rgb.at<cv::Vec3b>(0, 0)[1],
            rgb.at<cv::Vec3b>(0, 0)[2]);
        cv::rectangle(
            drawn_image, cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
            cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)), color, 2);

        std::string label = std::string(COCO_CLASSES[class_id]) + ": " + cv::format("%.2f", det.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        int label_x = static_cast<int>(det.x1);
        int label_y = static_cast<int>(det.y1) - 5;
        if (label_y < text_size.height)
            label_y = static_cast<int>(det.y1) + text_size.height + 5;

        cv::rectangle(
            drawn_image, cv::Point(label_x, label_y - text_size.height - baseline),
            cv::Point(label_x + text_size.width, label_y + baseline), color, cv::FILLED);
        int brightness = static_cast<int>((color[0] + color[1] + color[2]) / 3);
        cv::Scalar text_color = brightness < 128 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);
        cv::putText(
            drawn_image, label, cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX,
            0.6, text_color, 1, cv::LINE_AA);
    }

    return drawn_image;
}
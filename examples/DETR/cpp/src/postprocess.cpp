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
#include <string>

const char *COCO_CLASSES[91] = {
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "doughnut",
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};

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

cv::Mat preprocess(const cv::Mat &img, std::tuple<int, int> new_shape)
{
    if (img.empty())
        return {};

    cv::Mat img_rgb;

    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGRA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else if (img.channels() == 1)
        cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2RGB);
    else
        return {};

    int input_height = std::get<0>(new_shape);
    int input_width = std::get<1>(new_shape);

    cv::Mat resized;
    cv::resize(img_rgb, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3);
    cv::subtract(float_img, cv::Scalar(123.675, 116.28, 103.53), float_img);
    cv::divide(float_img, cv::Scalar(58.395, 57.12, 57.375), float_img);

    return float_img;
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
            float value = std::round(src_ptr[i] / attr.scale + attr.zp);
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());

        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale + attr.zp);
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, value)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());

        for (int i = 0; i < total_elements; ++i)
        {
            float value = std::round(src_ptr[i] / attr.scale + attr.zp);
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, value)));
        }
    }
    else
    {
        std::cerr << "prepare_input_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

std::vector<Detection> postprocess(const std::vector<float *> &out_ptrs,
                                   const std::vector<std::vector<int>> &out_shapes,
                                   int image_width, int image_height, float conf_thresh)
{
    std::vector<Detection> detections;

    if (out_ptrs.size() < 2 || out_shapes.size() < 2)
        return detections;

    const std::vector<int> &logits_shape = out_shapes[0];
    const std::vector<int> &boxes_shape = out_shapes[1];

    if (logits_shape.size() != 2 || boxes_shape.size() != 2)
    {
        std::cerr << "Unexpected DETR output shapes" << std::endl;
        return detections;
    }

    int num_queries = logits_shape[0];
    int num_logits = logits_shape[1];
    int num_classes = num_logits - 1;

    if (boxes_shape[0] != num_queries || boxes_shape[1] != 4)
    {
        std::cerr << "DETR logits and box query counts do not match" << std::endl;
        return detections;
    }

    const float *logits = out_ptrs[0];
    const float *boxes = out_ptrs[1];

    for (int query = 0; query < num_queries; ++query)
    {
        const float *query_logits = logits + query * num_logits;
        const float *query_box = boxes + query * 4;

        float max_logit = query_logits[0];

        for (int class_id = 1; class_id < num_logits; ++class_id)
            max_logit = std::max(max_logit, query_logits[class_id]);

        float exp_sum = 0.0f;

        for (int class_id = 0; class_id < num_logits; ++class_id)
            exp_sum += std::exp(query_logits[class_id] - max_logit);

        int best_class = 0;
        float best_logit = query_logits[0];

        for (int class_id = 1; class_id < num_classes; ++class_id)
        {
            if (query_logits[class_id] > best_logit)
            {
                best_logit = query_logits[class_id];
                best_class = class_id;
            }
        }

        float score = std::exp(best_logit - max_logit) / exp_sum;

        if (score < conf_thresh || best_class < 0 || best_class >= 91)
            continue;

        if (std::string(COCO_CLASSES[best_class]) == "N/A")
            continue;

        float center_x = query_box[0];
        float center_y = query_box[1];
        float width = query_box[2];
        float height = query_box[3];

        float x1 = (center_x - width * 0.5f) * image_width;
        float y1 = (center_y - height * 0.5f) * image_height;
        float x2 = (center_x + width * 0.5f) * image_width;
        float y2 = (center_y + height * 0.5f) * image_height;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(image_width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(image_height - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(image_width - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(image_height - 1));

        if (x2 <= x1 || y2 <= y1)
            continue;

        detections.push_back({x1, y1, x2, y2, score, best_class});
    }

    std::sort(detections.begin(), detections.end(), [](const Detection &a, const Detection &b)
              { return a.score > b.score; });

    return detections;
}

cv::Mat draw_detections(const cv::Mat &image, const std::vector<Detection> &detections)
{
    cv::Mat drawn_image = image.clone();

    for (const auto &det : detections)
    {
        if (det.class_id < 0 || det.class_id >= 91)
            continue;

        float hue = std::fmod(det.class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
        cv::Scalar color(pixel[0], pixel[1], pixel[2]);

        cv::rectangle(drawn_image, cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)),
                      cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)), color, 2);

        std::string label = std::string(COCO_CLASSES[det.class_id]) + ": " + cv::format("%.2f", det.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        int label_x = static_cast<int>(det.x1);
        int label_y = static_cast<int>(det.y1) - 5;

        if (label_y < text_size.height)
            label_y = static_cast<int>(det.y1) + text_size.height + 5;

        cv::rectangle(drawn_image, cv::Point(label_x, label_y - text_size.height - baseline),
                      cv::Point(label_x + text_size.width, label_y + baseline), color, cv::FILLED);

        int brightness = static_cast<int>((color[0] + color[1] + color[2]) / 3.0);
        cv::Scalar text_color = brightness < 128 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);

        cv::putText(drawn_image, label, cv::Point(label_x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv::LINE_AA);
    }

    return drawn_image;
}
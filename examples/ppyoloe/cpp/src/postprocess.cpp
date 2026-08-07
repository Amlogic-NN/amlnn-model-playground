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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>

const int STRIDES[3] = {8, 16, 32};
const float MEAN[3] = {123.675f, 116.28f, 103.53f};
const float STD[3] = {58.395f, 57.12f, 57.375f};

const char *COCO_CLASSES[NUM_CLASSES] = {
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

static float sigmoid(float value)
{
    value = std::max(-80.0f, std::min(80.0f, value));
    return 1.0f / (1.0f + std::exp(-value));
}

static float compute_iou(const Detection &det1, const Detection &det2)
{
    float x1 = std::max(det1.x1, det2.x1);
    float y1 = std::max(det1.y1, det2.y1);
    float x2 = std::min(det1.x2, det2.x2);
    float y2 = std::min(det1.y2, det2.y2);
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = std::max(0.0f, det1.x2 - det1.x1) * std::max(0.0f, det1.y2 - det1.y1);
    float area2 = std::max(0.0f, det2.x2 - det2.x1) * std::max(0.0f, det2.y2 - det2.y1);
    float union_area = area1 + area2 - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static std::vector<Detection> nms_by_class(
    const std::vector<Detection> &detections, float iou_threshold
)
{
    std::vector<Detection> final_detections;
    std::unordered_map<int, std::vector<Detection>> class_detections;
    for (const auto &detection : detections)
        class_detections[detection.class_id].push_back(detection);

    for (auto &[class_id, class_dets] : class_detections)
    {
        std::sort(
            class_dets.begin(), class_dets.end(),
            [](const Detection &a, const Detection &b) { return a.score > b.score; }
        );
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

    std::sort(
        final_detections.begin(), final_detections.end(),
        [](const Detection &a, const Detection &b) { return a.score > b.score; }
    );
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

std::tuple<cv::Mat, float, float> preprocess(
    cv::Mat img, std::tuple<int, int> new_shape
)
{
    if (img.empty())
        return {};

    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);
    float scale_x = static_cast<float>(target_w) / img.cols;
    float scale_y = static_cast<float>(target_h) / img.rows;

    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat img_rgb;
    if (img_resized.channels() == 4)
        cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGRA2RGB);
    else if (img_resized.channels() == 3)
        cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img_resized.clone();

    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32F);

    // PaddleDetection uses ImageNet RGB normalization after direct resize.
    for (int y = 0; y < target_h; ++y)
    {
        cv::Vec3f *row = img_float.ptr<cv::Vec3f>(y);
        for (int x = 0; x < target_w; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
                row[x][channel] = (row[x][channel] - MEAN[channel]) / STD[channel];
        }
    }

    return std::make_tuple(img_float, scale_x, scale_y);
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
    const std::vector<float *> &out_ptrs,
    const std::vector<std::vector<int>> &out_shapes,
    int input_h, int input_w, float scale_x, float scale_y,
    float conf_thresh, float iou_threshold, int reg_max
)
{
    if (out_ptrs.size() != 6 || out_shapes.size() != 6)
    {
        std::cerr << "Expected exactly 6 PP-YOLOE outputs." << std::endl;
        return {};
    }

    float safe_threshold = std::max(1e-5f, std::min(conf_thresh, 1.0f - 1e-5f));
    float inverse_threshold = std::log(safe_threshold / (1.0f - safe_threshold));
    int num_bins = reg_max + 1;
    std::vector<Detection> detections;

    for (int output_idx = 0; output_idx < 3; ++output_idx)
    {
        int stride = STRIDES[output_idx];
        int grid_h = input_h / stride;
        int grid_w = input_w / stride;
        int num_cells = grid_h * grid_w;
        int dfl_idx = output_idx * 2;
        int class_idx = dfl_idx + 1;
        const auto &dfl_shape = out_shapes[dfl_idx];
        const auto &class_shape = out_shapes[class_idx];

        if (dfl_shape.size() != 3 || dfl_shape[0] != num_cells ||
            dfl_shape[1] != num_bins || dfl_shape[2] != 4)
        {
            std::cerr << "Unexpected DFL shape for stride " << stride << "." << std::endl;
            return {};
        }
        if (class_shape.size() != 3 || class_shape[0] != grid_h ||
            class_shape[1] != grid_w || class_shape[2] != NUM_CLASSES)
        {
            std::cerr << "Unexpected class shape for stride " << stride << "." << std::endl;
            return {};
        }

        float *dfl_data = out_ptrs[dfl_idx];
        float *class_data = out_ptrs[class_idx];

        for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx)
        {
            const float *cell_classes = class_data + cell_idx * NUM_CLASSES;
            float max_class_logit = -1e9f;
            int detected_class = -1;

            for (int class_id = 0; class_id < NUM_CLASSES; ++class_id)
            {
                if (cell_classes[class_id] > max_class_logit)
                {
                    max_class_logit = cell_classes[class_id];
                    detected_class = class_id;
                }
            }

            if (max_class_logit <= inverse_threshold)
                continue;

            float distances[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int side = 0; side < 4; ++side)
            {
                float max_dfl = -1e9f;
                for (int bin = 0; bin < num_bins; ++bin)
                {
                    int index = (cell_idx * num_bins + bin) * 4 + side;
                    max_dfl = std::max(max_dfl, dfl_data[index]);
                }

                float sum_exp = 0.0f;
                float dot_product = 0.0f;
                for (int bin = 0; bin < num_bins; ++bin)
                {
                    int index = (cell_idx * num_bins + bin) * 4 + side;
                    float exp_value = std::exp(dfl_data[index] - max_dfl);
                    sum_exp += exp_value;
                    dot_product += exp_value * bin;
                }
                distances[side] = dot_product / sum_exp;
            }

            int gx = cell_idx % grid_w;
            int gy = cell_idx / grid_w;
            float center_x = (gx + 0.5f) * stride;
            float center_y = (gy + 0.5f) * stride;
            float x1 = (center_x - distances[0] * stride) / scale_x;
            float y1 = (center_y - distances[1] * stride) / scale_y;
            float x2 = (center_x + distances[2] * stride) / scale_x;
            float y2 = (center_y + distances[3] * stride) / scale_y;
            detections.push_back({
                std::max(0.0f, x1), std::max(0.0f, y1),
                std::max(0.0f, x2), std::max(0.0f, y2),
                sigmoid(max_class_logit), detected_class
            });
        }
    }

    return nms_by_class(detections, iou_threshold);
}

cv::Mat draw_detections(cv::Mat image, const std::vector<Detection> &detections)
{
    cv::Mat drawn_image = image.clone();

    for (const auto &detection : detections)
    {
        int class_id = detection.class_id;
        if (class_id < 0 || class_id >= NUM_CLASSES)
            continue;

        float hue = std::fmod(class_id * 137.508f, 360.0f);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
        cv::Scalar color(pixel[0], pixel[1], pixel[2]);

        int x1 = static_cast<int>(detection.x1);
        int y1 = static_cast<int>(detection.y1);
        int x2 = static_cast<int>(detection.x2);
        int y2 = static_cast<int>(detection.y2);
        cv::rectangle(drawn_image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string label = std::string(COCO_CLASSES[class_id]) +
            ": " + cv::format("%.2f", detection.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline
        );
        int label_x = std::max(0, x1);
        int label_y = std::max(y1, text_size.height + 10);
        cv::rectangle(
            drawn_image, cv::Point(label_x, label_y - text_size.height - 10),
            cv::Point(label_x + text_size.width, label_y), color, cv::FILLED
        );

        int brightness = (pixel[0] + pixel[1] + pixel[2]) / 3;
        cv::Scalar text_color = brightness < 128
                                    ? cv::Scalar(255, 255, 255)
                                    : cv::Scalar(0, 0, 0);
        cv::putText(
            drawn_image, label, cv::Point(label_x, label_y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv::LINE_AA
        );
    }

    return drawn_image;
}
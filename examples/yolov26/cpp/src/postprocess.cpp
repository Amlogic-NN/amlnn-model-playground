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
#include <vector>

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

const int STRIDES[3] = {8, 16, 32};
const int MAX_DETECTIONS = 300;

struct LocationCandidate
{
    float max_raw_score;
    const float *class_data;
    const float *bbox_data;
    int x;
    int y;
    int stride;
    int num_classes;
};

struct ClassCandidate
{
    float raw_score;
    int location_index;
    int class_id;
};

static float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
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

std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape)
{
    cv::Mat img_rgb;
    if (img.empty())
        return {};

    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img;

    int orig_h = img.rows;
    int orig_w = img.cols;
    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);

    float scale = std::min(static_cast<float>(target_h) / orig_h,
                           static_cast<float>(target_w) / orig_w);

    int new_h = static_cast<int>(std::round(orig_h * scale));
    int new_w = static_cast<int>(std::round(orig_w * scale));

    cv::Mat img_resized;
    if (new_h == orig_h && new_w == orig_w)
        img_resized = img_rgb;
    else
        cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_h = target_h - new_h;
    int pad_w = target_w - new_w;
    int pad_left = static_cast<int>(std::round(pad_w / 2.0 - 0.1));
    int pad_right = static_cast<int>(std::round(pad_w / 2.0 + 0.1));
    int pad_top = static_cast<int>(std::round(pad_h / 2.0 - 0.1));
    int pad_bottom = static_cast<int>(std::round(pad_h / 2.0 + 0.1));

    cv::Mat img_padded;
    if (pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0)
    {
        img_padded = img_resized;
    }
    else
    {
        cv::copyMakeBorder(
            img_resized, img_padded,
            pad_top, pad_bottom, pad_left, pad_right,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    }

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

std::vector<Detection> postprocess(const std::vector<float *> &out_ptrs,
                                   const std::vector<std::vector<int>> &out_shapes,
                                   int input_h, int input_w,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh)
{
    if (out_ptrs.size() != 6 || out_shapes.size() != 6)
    {
        std::cerr << "Expected 6 YOLO26 outputs." << std::endl;
        return {};
    }

    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));

    std::vector<LocationCandidate> locations;
    locations.reserve(
        input_h / STRIDES[0] * input_w / STRIDES[0] +
        input_h / STRIDES[1] * input_w / STRIDES[1] +
        input_h / STRIDES[2] * input_w / STRIDES[2]);

    for (int s = 0; s < 3; ++s)
    {
        int cls_idx = s * 2;
        int bbox_idx = s * 2 + 1;
        int stride = STRIDES[s];

        float *cls_data = out_ptrs[cls_idx];
        float *bbox_data = out_ptrs[bbox_idx];
        const auto &cls_shape = out_shapes[cls_idx];
        const auto &bbox_shape = out_shapes[bbox_idx];

        if (cls_shape.size() != 3)
        {
            std::cerr << "Expected NHWC class output [H, W, C] for output "
                      << cls_idx << ", got " << cls_shape.size() << " dimensions." << std::endl;
            return {};
        }

        if (bbox_shape.size() != 3)
        {
            std::cerr << "Expected NHWC bbox output [H, W, 4] for output "
                      << bbox_idx << ", got " << bbox_shape.size() << " dimensions." << std::endl;
            return {};
        }

        int height = cls_shape[0];
        int width = cls_shape[1];
        int num_classes = cls_shape[2];

        if (bbox_shape[0] != height || bbox_shape[1] != width || bbox_shape[2] != 4)
        {
            std::cerr << "BBox output " << bbox_idx << " shape does not match ["
                      << height << ", " << width << ", 4]." << std::endl;
            return {};
        }

        if (height != input_h / stride || width != input_w / stride)
        {
            std::cerr << "Output grid " << height << "x" << width
                      << " does not match stride " << stride << " for input "
                      << input_h << "x" << input_w << "." << std::endl;
            return {};
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int cell_idx = y * width + x;
                const float *class_data = cls_data + cell_idx * num_classes;
                const float *cell_bbox_data = bbox_data + cell_idx * 4;

                float max_raw_score = class_data[0];
                for (int c = 1; c < num_classes; ++c)
                    max_raw_score = std::max(max_raw_score, class_data[c]);

                locations.push_back({
                    max_raw_score,
                    class_data,
                    cell_bbox_data,
                    x,
                    y,
                    stride,
                    num_classes});
            }
        }
    }

    int location_count = std::min(MAX_DETECTIONS, static_cast<int>(locations.size()));
    if (location_count == 0)
        return {};

    std::partial_sort(
        locations.begin(),
        locations.begin() + location_count,
        locations.end(),
        [](const LocationCandidate &a, const LocationCandidate &b)
        {
            return a.max_raw_score > b.max_raw_score;
        });
    locations.resize(location_count);

    std::vector<ClassCandidate> class_candidates;
    for (int location_idx = 0; location_idx < location_count; ++location_idx)
    {
        const auto &location = locations[location_idx];
        for (int class_id = 0; class_id < location.num_classes; ++class_id)
        {
            class_candidates.push_back({
                location.class_data[class_id],
                location_idx,
                class_id});
        }
    }

    int detection_count = std::min(MAX_DETECTIONS, static_cast<int>(class_candidates.size()));
    std::partial_sort(
        class_candidates.begin(),
        class_candidates.begin() + detection_count,
        class_candidates.end(),
        [](const ClassCandidate &a, const ClassCandidate &b)
        {
            return a.raw_score > b.raw_score;
        });
    class_candidates.resize(detection_count);

    float safe_thresh = std::max(1e-5f, std::min(conf_thresh, 1.0f - 1e-5f));
    float inv_thresh = std::log(safe_thresh / (1.0f - safe_thresh));

    std::vector<Detection> detections;
    detections.reserve(detection_count);

    for (const auto &candidate : class_candidates)
    {
        if (candidate.raw_score <= inv_thresh)
            continue;

        const auto &location = locations[candidate.location_index];
        float left = location.bbox_data[0];
        float top = location.bbox_data[1];
        float right = location.bbox_data[2];
        float bottom = location.bbox_data[3];

        float anchor_x = static_cast<float>(location.x) + 0.5f;
        float anchor_y = static_cast<float>(location.y) + 0.5f;

        float x1 = ((anchor_x - left) * location.stride - pad_left) / scale;
        float y1 = ((anchor_y - top) * location.stride - pad_top) / scale;
        float x2 = ((anchor_x + right) * location.stride - pad_left) / scale;
        float y2 = ((anchor_y + bottom) * location.stride - pad_top) / scale;

        detections.push_back({
            std::max(0.0f, x1),
            std::max(0.0f, y1),
            std::max(0.0f, x2),
            std::max(0.0f, y2),
            sigmoid(candidate.raw_score),
            candidate.class_id});
    }

    return detections;
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
        cv::Scalar color(rgb.at<cv::Vec3b>(0, 0)[0], rgb.at<cv::Vec3b>(0, 0)[1], rgb.at<cv::Vec3b>(0, 0)[2]);

        cv::rectangle(
            drawn_image,
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

        cv::rectangle(
            drawn_image,
            cv::Point(label_x, label_y - text_size.height - baseline),
            cv::Point(label_x + text_size.width, label_y + baseline),
            color, cv::FILLED);

        int brightness = static_cast<int>((color[0] + color[1] + color[2]) / 3);
        cv::Scalar text_color = brightness < 128
                                    ? cv::Scalar(255, 255, 255)
                                    : cv::Scalar(0, 0, 0);

        cv::putText(
            drawn_image, label, cv::Point(label_x, label_y),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv::LINE_AA);
    }

    return drawn_image;
}
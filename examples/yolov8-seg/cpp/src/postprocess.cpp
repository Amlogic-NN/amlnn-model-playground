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
#include <unordered_map>

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
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

const int STRIDES[3] = {8, 16, 32};

static float sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

static cv::Scalar get_color(int class_id)
{
    float hue = std::fmod(class_id * 137.508f, 360.0f);
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue / 2.0f, 204, 230));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
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
    const std::vector<Detection> &detections, float iou_threshold)
{
    if (detections.empty())
        return {};

    std::vector<Detection> final_detections;
    std::unordered_map<int, std::vector<Detection>> class_detections;
    for (const auto &detection : detections)
        class_detections[detection.class_id].push_back(detection);

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

    std::sort(
        final_detections.begin(), final_detections.end(),
        [](const Detection &a, const Detection &b)
        { return a.score > b.score; });
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
    std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
    float conf_thresh, float iou_threshold, int reg_max)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));
    std::vector<Detection> detections_orig;

    const auto &prototype_shape = out_shapes[9];
    if ((prototype_shape.size() == 4 &&
         (prototype_shape[1] != 160 || prototype_shape[2] != 160 || prototype_shape[3] != NUM_MASK_COEFFICIENTS)) ||
        (prototype_shape.size() == 3 &&
         (prototype_shape[0] != 160 || prototype_shape[1] != 160 || prototype_shape[2] != NUM_MASK_COEFFICIENTS)) ||
        (prototype_shape.size() != 3 && prototype_shape.size() != 4))
    {
        std::cerr << "Unexpected prototype output shape." << std::endl;
        return {};
    }

    float safe_thresh = std::max(1e-5f, std::min(conf_thresh, 1.0f - 1e-5f));
    float inv_thresh = std::log(safe_thresh / (1.0f - safe_thresh));
    int dfl_channels = 4 * reg_max;

    // Output order: [DFL_8, CLS_8, MASK_8, DFL_16, CLS_16, MASK_16, DFL_32, CLS_32, MASK_32, PROTO].
    for (int s = 0; s < 3; ++s)
    {
        int dfl_idx = s * 3;
        int class_idx = s * 3 + 1;
        int mask_idx = s * 3 + 2;
        int stride = STRIDES[s];

        float *dfl_data = out_ptrs[dfl_idx];
        float *class_data = out_ptrs[class_idx];
        float *mask_data = out_ptrs[mask_idx];

        const auto &dfl_shape = out_shapes[dfl_idx];
        const auto &class_shape = out_shapes[class_idx];
        const auto &mask_shape = out_shapes[mask_idx];

        int height = 1;
        int width = 1;
        int channels = 1;

        if (dfl_shape.size() == 4)
        {
            height = dfl_shape[1];
            width = dfl_shape[2];
            channels = dfl_shape[3];
        }
        else if (dfl_shape.size() == 3)
        {
            height = dfl_shape[0];
            width = dfl_shape[1];
            channels = dfl_shape[2];
        }
        else
        {
            std::cerr << "Unexpected DFL output shape for output " << dfl_idx << std::endl;
            continue;
        }

        if (channels != dfl_channels)
        {
            std::cerr << "DFL output " << dfl_idx << " expected " << dfl_channels << " channels, got " << channels << std::endl;
            continue;
        }

        int class_height = 1;
        int class_width = 1;
        int class_channels = 1;

        if (class_shape.size() == 4)
        {
            class_height = class_shape[1];
            class_width = class_shape[2];
            class_channels = class_shape[3];
        }
        else if (class_shape.size() == 3)
        {
            class_height = class_shape[0];
            class_width = class_shape[1];
            class_channels = class_shape[2];
        }
        else
        {
            std::cerr << "Unexpected class output shape for output " << class_idx << std::endl;
            continue;
        }

        int mask_height = 1;
        int mask_width = 1;
        int mask_channels = 1;

        if (mask_shape.size() == 4)
        {
            mask_height = mask_shape[1];
            mask_width = mask_shape[2];
            mask_channels = mask_shape[3];
        }
        else if (mask_shape.size() == 3)
        {
            mask_height = mask_shape[0];
            mask_width = mask_shape[1];
            mask_channels = mask_shape[2];
        }
        else
        {
            std::cerr << "Unexpected mask coefficient output shape for output " << mask_idx << std::endl;
            continue;
        }

        if (class_height != height || class_width != width || class_channels != NUM_CLASSES)
        {
            std::cerr << "Class output " << class_idx << " does not match [" << height << ", " << width << ", " << NUM_CLASSES << "]" << std::endl;
            continue;
        }

        if (mask_height != height || mask_width != width || mask_channels != NUM_MASK_COEFFICIENTS)
        {
            std::cerr << "Mask output " << mask_idx << " does not match [" << height << ", " << width << ", " << NUM_MASK_COEFFICIENTS << "]" << std::endl;
            continue;
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int cell_idx = y * width + x;
                const float *class_cell = class_data + cell_idx * NUM_CLASSES;
                float max_raw_score = -1e9f;
                int class_id = -1;

                for (int c = 0; c < NUM_CLASSES; ++c)
                {
                    if (class_cell[c] > max_raw_score)
                    {
                        max_raw_score = class_cell[c];
                        class_id = c;
                    }
                }

                // Skip DFL and mask coefficient decoding for low-confidence cells.
                if (max_raw_score <= inv_thresh)
                    continue;

                const float *dfl_cell = dfl_data + cell_idx * dfl_channels;
                float distances[4] = {};

                // Decode DFL in left, top, right, bottom order.
                for (int side = 0; side < 4; ++side)
                {
                    const float *distribution = dfl_cell + side * reg_max;
                    float max_value = distribution[0];

                    for (int i = 1; i < reg_max; ++i)
                        max_value = std::max(max_value, distribution[i]);

                    float sum = 0.0f;
                    float weighted_sum = 0.0f;

                    for (int i = 0; i < reg_max; ++i)
                    {
                        float probability = std::exp(distribution[i] - max_value);
                        sum += probability;
                        weighted_sum += probability * static_cast<float>(i);
                    }

                    distances[side] = weighted_sum / sum;
                }

                float center_x = (static_cast<float>(x) + 0.5f) * stride;
                float center_y = (static_cast<float>(y) + 0.5f) * stride;

                Detection detection;
                detection.x1 = (center_x - distances[0] * stride - pad_left) / scale;
                detection.y1 = (center_y - distances[1] * stride - pad_top) / scale;
                detection.x2 = (center_x + distances[2] * stride - pad_left) / scale;
                detection.y2 = (center_y + distances[3] * stride - pad_top) / scale;
                detection.score = sigmoid(max_raw_score);
                detection.class_id = class_id;

                if (detection.x2 <= detection.x1 || detection.y2 <= detection.y1)
                    continue;

                const float *mask_cell = mask_data + cell_idx * NUM_MASK_COEFFICIENTS;
                for (int coefficient_idx = 0; coefficient_idx < NUM_MASK_COEFFICIENTS; ++coefficient_idx)
                    detection.mask_coefficients[coefficient_idx] = mask_cell[coefficient_idx];

                detections_orig.push_back(detection);
            }
        }
    }

    return nms_by_class(detections_orig, iou_threshold);
}

cv::Mat draw_detections(
    cv::Mat image, const std::vector<Detection> &detections,
    float *prototype_data, const std::vector<int> &prototype_shape,
    int input_h, int input_w, float scale, std::tuple<int, int> pad, float alpha)
{
    cv::Mat drawn_image = image.clone();
    if (prototype_shape.size() != 3 || prototype_shape[0] != 160 ||
        prototype_shape[1] != 160 || prototype_shape[2] != NUM_MASK_COEFFICIENTS)
    {
        std::cerr << "Cannot draw masks: unexpected prototype mask shape." << std::endl;
        return drawn_image;
    }

    int prototype_h = prototype_shape[0];
    int prototype_w = prototype_shape[1];
    int pad_left = std::get<0>(pad);
    int pad_top = std::get<1>(pad);
    int resized_w = static_cast<int>(std::round(image.cols * scale));
    int resized_h = static_cast<int>(std::round(image.rows * scale));
    cv::Mat prototype_mat(
        prototype_h * prototype_w, NUM_MASK_COEFFICIENTS, CV_32F, prototype_data);

    for (const auto &detection : detections)
    {
        int class_id = detection.class_id;
        if (class_id < 0 || class_id >= NUM_CLASSES)
            continue;

        cv::Scalar color = get_color(class_id);

        // Contract 32 mask coefficients against the NHWC prototype channels.
        cv::Mat coefficient_mat(
            1, NUM_MASK_COEFFICIENTS, CV_32F,
            const_cast<float *>(detection.mask_coefficients.data()));
        cv::Mat raw_mask = coefficient_mat * prototype_mat.t();
        raw_mask = raw_mask.reshape(1, prototype_h);

        cv::Mat clipped_mask;
        cv::max(raw_mask, -80.0, clipped_mask);
        cv::min(clipped_mask, 80.0, clipped_mask);
        cv::Mat exponential;
        cv::exp(-clipped_mask, exponential);
        cv::Mat sigmoid_mask;
        cv::divide(1.0, 1.0 + exponential, sigmoid_mask);

        cv::Mat input_mask;
        cv::resize(
            sigmoid_mask, input_mask, cv::Size(input_w, input_h),
            0, 0, cv::INTER_LINEAR);

        int roi_x = std::max(0, pad_left);
        int roi_y = std::max(0, pad_top);
        int roi_width = std::min(resized_w, input_w - roi_x);
        int roi_height = std::min(resized_h, input_h - roi_y);

        if (roi_width > 0 && roi_height > 0)
        {
            cv::Mat cropped_mask = input_mask(cv::Rect(roi_x, roi_y, roi_width, roi_height));
            cv::Mat final_mask;
            cv::resize(
                cropped_mask, final_mask, cv::Size(image.cols, image.rows),
                0, 0, cv::INTER_LINEAR);

            cv::Mat binary_mask = final_mask > 0.5f;

            // Crop the reconstructed mask to its retained detection box.
            int box_x1 = std::max(0, static_cast<int>(std::floor(detection.x1)));
            int box_y1 = std::max(0, static_cast<int>(std::floor(detection.y1)));
            int box_x2 = std::min(image.cols, static_cast<int>(std::ceil(detection.x2)));
            int box_y2 = std::min(image.rows, static_cast<int>(std::ceil(detection.y2)));
            cv::Mat bbox_mask = cv::Mat::zeros(image.size(), CV_8U);
            if (box_x2 > box_x1 && box_y2 > box_y1)
            {
                cv::Rect box_roi(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1);
                bbox_mask(box_roi).setTo(255);
            }
            cv::bitwise_and(binary_mask, bbox_mask, binary_mask);

            if (cv::countNonZero(binary_mask) > 0)
            {
                cv::Mat colored_mask = cv::Mat::zeros(image.size(), image.type());
                colored_mask.setTo(color, binary_mask);
                cv::Mat blended;
                cv::addWeighted(drawn_image, 1.0f - alpha, colored_mask, alpha, 0, blended);
                blended.copyTo(drawn_image, binary_mask);
            }
        }

        cv::rectangle(
            drawn_image,
            cv::Point(static_cast<int>(detection.x1), static_cast<int>(detection.y1)),
            cv::Point(static_cast<int>(detection.x2), static_cast<int>(detection.y2)),
            color, 2);

        std::string label = std::string(COCO_CLASSES[class_id]) +
                            ": " + cv::format("%.2f", detection.score);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        int label_x = std::max(0, static_cast<int>(detection.x1));
        int label_y = std::max(
            static_cast<int>(detection.y1) - 5, text_size.height + baseline);
        cv::rectangle(
            drawn_image, cv::Point(label_x, label_y - text_size.height - baseline),
            cv::Point(label_x + text_size.width, label_y + baseline), color, cv::FILLED);

        int brightness = static_cast<int>((color[0] + color[1] + color[2]) / 3);
        cv::Scalar text_color = brightness < 128
                                    ? cv::Scalar(255, 255, 255)
                                    : cv::Scalar(0, 0, 0);
        cv::putText(
            drawn_image, label, cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX,
            0.6, text_color, 1, cv::LINE_AA);
    }

    return drawn_image;
}
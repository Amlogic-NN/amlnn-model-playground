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
#include <fstream>
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

const int STRIDES[3] = {8, 16, 32};

static float sigmoid(float value)
{
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

static std::vector<Detection> nms_by_class(const std::vector<Detection> &detections, float iou_threshold)
{
    if (detections.empty())
        return {};

    std::vector<Detection> final_detections;
    std::unordered_map<int, std::vector<Detection>> class_detections;
    for (const auto &det : detections)
        class_detections[det.class_id].push_back(det);

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
                if (!removed[j] && compute_iou(cls_dets[i], cls_dets[j]) > iou_threshold)
                    removed[j] = true;
            }
        }
    }

    return final_detections;
}

static float decode_dfl_distance(const float *dfl_data, int cell_idx, int side, int reg_max)
{
    const float *side_logits = dfl_data + cell_idx * 4 * reg_max + side * reg_max;

    float max_logit = side_logits[0];
    for (int bin = 1; bin < reg_max; ++bin)
        max_logit = std::max(max_logit, side_logits[bin]);

    float weighted_sum = 0.0f;
    float exp_sum = 0.0f;

    for (int bin = 0; bin < reg_max; ++bin)
    {
        float probability = std::exp(side_logits[bin] - max_logit);
        weighted_sum += probability * static_cast<float>(bin);
        exp_sum += probability;
    }

    return exp_sum > 0.0f ? weighted_sum / exp_sum : 0.0f;
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

cv::Mat load_image(const std::string &path, int input_height, int input_width)
{
    std::string extension = path.substr(path.find_last_of('.'));
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp")
    {
        cv::Mat image = cv::imread(path);
        if (image.empty())
            std::cerr << "Failed to read image: " << path << std::endl;
        return image;
    }

    if (extension == ".txt")
    {
        std::ifstream file(path);
        if (!file)
        {
            std::cerr << "Failed to open TXT image: " << path << std::endl;
            return {};
        }

        cv::Mat image(input_height, input_width, CV_8UC3);
        size_t expected_size = static_cast<size_t>(input_height) * input_width * 3;

        for (size_t i = 0; i < expected_size; ++i)
        {
            int value;
            if (!(file >> value))
            {
                std::cerr << "Invalid TXT image data size: expected " << expected_size
                          << " values for " << input_height << "x" << input_width << "x3: " << path << std::endl;
                return {};
            }

            if (value < 0 || value > 255)
            {
                std::cerr << "TXT image pixel value outside [0, 255]: " << path << std::endl;
                return {};
            }

            image.data[i] = static_cast<uint8_t>(value);
        }

        int extra_value;
        if (file >> extra_value)
        {
            std::cerr << "TXT image contains unexpected extra data: " << path << std::endl;
            return {};
        }

        return image;
    }

    return {};
}

static size_t get_tensor_type_size(int tensor_type)
{
    if (tensor_type == AMLNN_TENSOR_FLOAT32)
        return sizeof(float);
    if (tensor_type == AMLNN_TENSOR_INT16)
        return sizeof(int16_t);
    if (tensor_type == AMLNN_TENSOR_INT8)
        return sizeof(int8_t);
    if (tensor_type == AMLNN_TENSOR_UINT8)
        return sizeof(uint8_t);

    return 0;
}

template <typename T>
static std::vector<uint8_t> pack_tensor_values(const std::vector<T> &values)
{
    std::vector<uint8_t> data(values.size() * sizeof(T));
    std::memcpy(data.data(), values.data(), data.size());
    return data;
}

std::vector<uint8_t> load_direct_input_tensor(const std::string &path, const amlnn_tensor_attr &attr,
                                              int input_height, int input_width)
{
    std::string extension = path.substr(path.find_last_of('.'));
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    size_t element_size = get_tensor_type_size(attr.type);
    if (element_size == 0)
    {
        std::cerr << "Unsupported direct input tensor type " << attr.type << std::endl;
        return {};
    }

    size_t expected_elements = static_cast<size_t>(input_height) * input_width * 3;
    size_t expected_bytes = expected_elements * element_size;

    if (extension == ".bin")
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cerr << "Failed to open BIN input: " << path << std::endl;
            return {};
        }

        size_t file_size = static_cast<size_t>(file.tellg());
        if (file_size != expected_bytes)
        {
            std::cerr << "Invalid BIN input size: expected " << expected_bytes
                      << " bytes, got " << file_size << ": " << path << std::endl;
            return {};
        }

        std::vector<uint8_t> data(expected_bytes);
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char *>(data.data()), expected_bytes);

        if (!file)
        {
            std::cerr << "Failed to read BIN input: " << path << std::endl;
            return {};
        }

        return data;
    }

    if (extension == ".qtxt")
    {
        std::ifstream file(path);
        if (!file)
        {
            std::cerr << "Failed to open QTXT input: " << path << std::endl;
            return {};
        }

        std::vector<uint8_t> data;

        if (attr.type == AMLNN_TENSOR_FLOAT32)
        {
            std::vector<float> values(expected_elements);
            for (size_t i = 0; i < expected_elements; ++i)
            {
                if (!(file >> values[i]))
                {
                    std::cerr << "Invalid QTXT input data size: expected " << expected_elements
                              << " values: " << path << std::endl;
                    return {};
                }
            }
            data = pack_tensor_values(values);
        }
        else if (attr.type == AMLNN_TENSOR_INT16)
        {
            std::vector<int16_t> values(expected_elements);
            for (size_t i = 0; i < expected_elements; ++i)
            {
                int value;
                if (!(file >> value))
                {
                    std::cerr << "Invalid QTXT input data size: expected " << expected_elements
                              << " values: " << path << std::endl;
                    return {};
                }
                if (value < -32768 || value > 32767)
                {
                    std::cerr << "QTXT int16 value outside [-32768, 32767]: " << path << std::endl;
                    return {};
                }
                values[i] = static_cast<int16_t>(value);
            }
            data = pack_tensor_values(values);
        }
        else if (attr.type == AMLNN_TENSOR_INT8)
        {
            std::vector<int8_t> values(expected_elements);
            for (size_t i = 0; i < expected_elements; ++i)
            {
                int value;
                if (!(file >> value))
                {
                    std::cerr << "Invalid QTXT input data size: expected " << expected_elements
                              << " values: " << path << std::endl;
                    return {};
                }
                if (value < -128 || value > 127)
                {
                    std::cerr << "QTXT int8 value outside [-128, 127]: " << path << std::endl;
                    return {};
                }
                values[i] = static_cast<int8_t>(value);
            }
            data = pack_tensor_values(values);
        }
        else if (attr.type == AMLNN_TENSOR_UINT8)
        {
            std::vector<uint8_t> values(expected_elements);
            for (size_t i = 0; i < expected_elements; ++i)
            {
                int value;
                if (!(file >> value))
                {
                    std::cerr << "Invalid QTXT input data size: expected " << expected_elements
                              << " values: " << path << std::endl;
                    return {};
                }
                if (value < 0 || value > 255)
                {
                    std::cerr << "QTXT uint8 value outside [0, 255]: " << path << std::endl;
                    return {};
                }
                values[i] = static_cast<uint8_t>(value);
            }
            data = pack_tensor_values(values);
        }

        double extra_value;
        if (file >> extra_value)
        {
            std::cerr << "QTXT input contains unexpected extra data: " << path << std::endl;
            return {};
        }

        return data;
    }

    return {};
}

cv::Mat reconstruct_direct_input_image(const std::vector<uint8_t> &tensor_data,
                                       const amlnn_tensor_attr &attr,
                                       int input_height, int input_width)
{
    size_t element_size = get_tensor_type_size(attr.type);
    size_t expected_elements = static_cast<size_t>(input_height) * input_width * 3;
    if (element_size == 0 || tensor_data.size() != expected_elements * element_size)
        return {};

    cv::Mat rgb_image(input_height, input_width, CV_8UC3);

    for (size_t i = 0; i < expected_elements; ++i)
    {
        float normalized_value = 0.0f;

        if (attr.type == AMLNN_TENSOR_FLOAT32)
        {
            float value;
            std::memcpy(&value, tensor_data.data() + i * sizeof(float), sizeof(float));
            normalized_value = value;
        }
        else if (attr.type == AMLNN_TENSOR_INT16)
        {
            int16_t value;
            std::memcpy(&value, tensor_data.data() + i * sizeof(int16_t), sizeof(int16_t));
            normalized_value = (static_cast<float>(value) - attr.zp) * attr.scale;
        }
        else if (attr.type == AMLNN_TENSOR_INT8)
        {
            int8_t value = static_cast<int8_t>(tensor_data[i]);
            normalized_value = (static_cast<float>(value) - attr.zp) * attr.scale;
        }
        else if (attr.type == AMLNN_TENSOR_UINT8)
        {
            uint8_t value = tensor_data[i];
            normalized_value = (static_cast<float>(value) - attr.zp) * attr.scale;
        }

        float pixel_value = std::max(0.0f, std::min(255.0f, normalized_value * 255.0f));
        rgb_image.data[i] = static_cast<uint8_t>(std::round(pixel_value));
    }

    cv::Mat bgr_image;
    cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);
    return bgr_image;
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
                                   float conf_thresh, float iou_threshold, int reg_max)
{
    float scale = std::get<1>(input_tuple);
    int pad_left = std::get<0>(std::get<2>(input_tuple));
    int pad_top = std::get<1>(std::get<2>(input_tuple));
    std::vector<Detection> detections_orig;

    float safe_thresh = std::max(1e-5f, std::min(conf_thresh, 1.0f - 1e-5f));
    float inv_thresh = std::log(safe_thresh / (1.0f - safe_thresh));

    for (int s = 0; s < 3; ++s)
    {
        int cls_idx = s * 2;
        int dfl_idx = s * 2 + 1;
        int stride = STRIDES[s];

        float *cls_data = out_ptrs[cls_idx];
        float *dfl_data = out_ptrs[dfl_idx];
        const auto &cls_shape = out_shapes[cls_idx];
        const auto &dfl_shape = out_shapes[dfl_idx];

        if (cls_shape.size() != 3)
        {
            std::cerr << "Expected NHWC class output [H, W, C] for output "
                      << cls_idx << ", got " << cls_shape.size() << " dimensions." << std::endl;
            continue;
        }

        if (dfl_shape.size() != 3)
        {
            std::cerr << "Expected NHWC DFL output [H, W, 4 * REG_MAX] for output "
                      << dfl_idx << ", got " << dfl_shape.size() << " dimensions." << std::endl;
            continue;
        }

        int height = cls_shape[0];
        int width = cls_shape[1];
        int num_classes = cls_shape[2];
        int dfl_channels = 4 * reg_max;

        if (dfl_shape[0] != height || dfl_shape[1] != width || dfl_shape[2] != dfl_channels)
        {
            std::cerr << "DFL output " << dfl_idx << " shape does not match ["
                      << height << ", " << width << ", " << dfl_channels << "]." << std::endl;
            continue;
        }

        if (height != input_h / stride || width != input_w / stride)
        {
            std::cerr << "Output grid " << height << "x" << width
                      << " does not match stride " << stride << " for input "
                      << input_h << "x" << input_w << "." << std::endl;
            continue;
        }

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int cell_idx = y * width + x;
                const float *class_data = cls_data + cell_idx * num_classes;

                float max_raw_score = -1e9f;
                int class_id = -1;

                for (int c = 0; c < num_classes; ++c)
                {
                    float value = class_data[c];
                    if (value > max_raw_score)
                    {
                        max_raw_score = value;
                        class_id = c;
                    }
                }

                if (max_raw_score <= inv_thresh)
                    continue;

                float left = decode_dfl_distance(dfl_data, cell_idx, 0, reg_max);
                float top = decode_dfl_distance(dfl_data, cell_idx, 1, reg_max);
                float right = decode_dfl_distance(dfl_data, cell_idx, 2, reg_max);
                float bottom = decode_dfl_distance(dfl_data, cell_idx, 3, reg_max);

                float center_x = (static_cast<float>(x) + 0.5f) * stride;
                float center_y = (static_cast<float>(y) + 0.5f) * stride;
                float x1 = (center_x - left * stride - pad_left) / scale;
                float y1 = (center_y - top * stride - pad_top) / scale;
                float x2 = (center_x + right * stride - pad_left) / scale;
                float y2 = (center_y + bottom * stride - pad_top) / scale;

                detections_orig.push_back({std::max(0.0f, x1),
                                           std::max(0.0f, y1),
                                           std::max(0.0f, x2),
                                           std::max(0.0f, y2),
                                           sigmoid(max_raw_score),
                                           class_id});
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
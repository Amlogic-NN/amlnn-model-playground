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
#include "clipper.h"
#include <algorithm>
#include <fstream>
#include <cmath>
#include <limits>

// ----- UTILS -----
std::vector<int> get_tensor_shape(amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
            shape.push_back(attr.dims[i]);
    }
    return shape;
}

std::vector<std::string> load_dict(const std::string &path)
{
    std::vector<std::string> dict;
    std::ifstream in(path);
    if (!in.is_open())
        return dict;
    std::string line;
    while (std::getline(in, line))
    {
        dict.push_back(line);
    }
    dict.push_back(" "); // Space character mapping
    return dict;
}

// ----- DET PIPELINE -----
std::tuple<cv::Mat, float> preprocess_det(const cv::Mat &image, const int width, const int height)
{
    if (image.empty())
        return std::make_tuple(cv::Mat(), 1.0f);

    // Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

    float ratio_max = std::max((float)rgb_img.cols / width, (float)rgb_img.rows / height);
    int new_w = std::min(int(rgb_img.cols / ratio_max), width);
    int new_h = std::min(int(rgb_img.rows / ratio_max), height);

    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(new_w, new_h));

    // Create padded image (black background)
    cv::Mat padded_img = cv::Mat::zeros(height, width, CV_8UC3);
    resized_img.copyTo(padded_img(cv::Rect(0, 0, new_w, new_h)));

    // Convert to Float and scale to [0, 1]
    cv::Mat float_img;
    padded_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // Apply ImageNet Normalization (Mean & Std)
    cv::Mat mean_mat(height, width, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std_mat(height, width, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));

    cv::subtract(float_img, mean_mat, float_img);
    cv::divide(float_img, std_mat, float_img);

    return std::make_tuple(float_img, ratio_max);
}

// std::vector<int8_t> quantize_input_det(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
// {
//     std::vector<int8_t> quantized_data(float_img.total() * float_img.channels());
//     const float *src = float_img.ptr<float>();

//     for (size_t i = 0; i < quantized_data.size(); ++i)
//     {
//         float q_val = std::round(src[i] / attr.scale) + attr.zp;
//         quantized_data[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, q_val)));
//     }
//     return quantized_data;
// }

std::vector<Object> postprocess_det(float *out, const std::vector<int> &shape, const cv::Mat &image, float box_score_thresh, float box_thresh, float scale)
{
    if (out == nullptr)
        return {};

    // Matches the updated 640 shape
    cv::Mat pred_map(DET_MODEL_HEIGHT, DET_MODEL_WIDTH, CV_32FC1, out);

    cv::Mat bit_map;
    bit_map = pred_map > box_thresh;
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(bit_map, bit_map, dila_ele, cv::Point(-1, -1), 1);

    return find_box(pred_map, bit_map, box_score_thresh, UNCLIP_RATIO, image, scale);
}

std::vector<Object> find_box(const cv::Mat pred_map, const cv::Mat &bit_map, const float box_score_thresh, const float unclip_ratio, const cv::Mat &image, float scale)
{
    std::vector<Object> res_boxes;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bit_map, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int num_coutours = contours.size() >= MAX_CANDIDATES ? MAX_CANDIDATES : contours.size();

    for (int i = 0; i < num_coutours; i++)
    {
        if (contours[i].size() <= 2)
            continue;

        float min_side_len, perimeter;
        Object text_box;

        std::vector<cv::Point> min_box = get_min_boxes(contours[i], min_side_len, perimeter);
        if (min_side_len < MIN_SIZE)
            continue;

        float score = get_box_score_fast(pred_map, contours[i]);
        if (score < box_score_thresh)
            continue;

        std::vector<cv::Point> clip_box = unclip(min_box, perimeter, unclip_ratio);
        std::vector<cv::Point> clip_min_box = get_min_boxes(clip_box, min_side_len, perimeter);
        if (min_side_len < MIN_SIZE + 2)
            continue;

        for (int j = 0; j < clip_min_box.size(); ++j)
        {
            clip_min_box[j].x = std::min(std::max(int(clip_min_box[j].x * scale), 0), image.cols);
            clip_min_box[j].y = std::min(std::max(int(clip_min_box[j].y * scale), 0), image.rows);
            text_box.box.push_back(clip_min_box[j]);
        }

        text_box.score = score;
        res_boxes.push_back(text_box);
    }
    return res_boxes;
}

std::vector<cv::Point> get_min_boxes(const std::vector<cv::Point> &in_vec, float &min_side_len, float &perimeter)
{
    std::vector<cv::Point> min_box_vec;
    cv::RotatedRect text_rect = cv::minAreaRect(in_vec);
    cv::Mat box_point2f;
    cv::boxPoints(text_rect, box_point2f);

    float *p1 = (float *)box_point2f.data;
    std::vector<cv::Point> temp_vec;

    for (int i = 0; i < 4; ++i, p1 += 2)
    {
        temp_vec.emplace_back(int(p1[0]), int(p1[1]));
    }

    std::sort(temp_vec.begin(), temp_vec.end(), cv_point_compare);

    int index1, index2, index3, index4;
    if (temp_vec[1].y > temp_vec[0].y)
    {
        index1 = 0;
        index4 = 1;
    }
    else
    {
        index1 = 1;
        index4 = 0;
    }

    if (temp_vec[3].y > temp_vec[2].y)
    {
        index2 = 2;
        index3 = 3;
    }
    else
    {
        index2 = 3;
        index3 = 2;
    }

    min_box_vec.push_back(temp_vec[index1]);
    min_box_vec.push_back(temp_vec[index2]);
    min_box_vec.push_back(temp_vec[index3]);
    min_box_vec.push_back(temp_vec[index4]);

    min_side_len = std::min(text_rect.size.width, text_rect.size.height);
    perimeter = 2.f * (text_rect.size.width + text_rect.size.height);
    return min_box_vec;
}

float get_box_score_fast(const cv::Mat &in_mat, const std::vector<cv::Point> &in_box)
{
    std::vector<cv::Point> box = in_box;
    int width = in_mat.cols;
    int height = in_mat.rows;

    int max_x = -1, max_y = -1;
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();

    for (int i = 0; i < box.size(); ++i)
    {
        if (max_x < box[i].x)
            max_x = box[i].x;
        if (max_y < box[i].y)
            max_y = box[i].y;
        if (min_x > box[i].x)
            min_x = box[i].x;
        if (min_y > box[i].y)
            min_y = box[i].y;
    }

    max_x = std::min(std::max(max_x, 0), width - 1);
    max_y = std::min(std::max(max_y, 0), height - 1);
    min_x = std::max(std::min(min_x, width - 1), 0);
    min_y = std::max(std::min(min_y, height - 1), 0);

    for (int i = 0; i < box.size(); ++i)
    {
        box[i].x -= min_x;
        box[i].y -= min_y;
    }

    std::vector<std::vector<cv::Point>> mask_box;
    mask_box.push_back(box);

    cv::Mat mask_mat(max_y - min_y + 1, max_x - min_x + 1, CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask_mat, mask_box, cv::Scalar(1), 1);

    return cv::mean(in_mat(cv::Rect(cv::Point(min_x, min_y), cv::Point(max_x + 1, max_y + 1))).clone(), mask_mat).val[0];
}

std::vector<cv::Point> unclip(const std::vector<cv::Point> &in_box, float perimeter, float unclip_ratio)
{
    std::vector<cv::Point> out_box;
    ClipperLib::Path poly;

    for (int i = 0; i < in_box.size(); ++i)
    {
        poly.push_back(ClipperLib::IntPoint(in_box[i].x, in_box[i].y));
    }

    double distance = unclip_ratio * ClipperLib::Area(poly) / (double)perimeter;

    ClipperLib::ClipperOffset clipper_offset;
    clipper_offset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths polys;
    polys.push_back(poly);
    clipper_offset.Execute(polys, distance);

    for (int i = 0; i < polys.size(); ++i)
    {
        for (int j = 0; j < polys[i].size(); ++j)
        {
            out_box.emplace_back(polys[i][j].X, polys[i][j].Y);
        }
    }
    return out_box;
}

bool cv_point_compare(const cv::Point &a, const cv::Point &b)
{
    return a.x < b.x;
}

// ----- REC PIPELINE -----
cv::Mat preprocess_rec(const cv::Mat &image, const int dest_width, const int dest_height)
{
    if (image.empty())
        return cv::Mat();

    cv::Mat rgb_img;
    cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

    float ratio = (float)rgb_img.cols / (float)rgb_img.rows;
    int resize_w = std::min(int(dest_height * ratio), dest_width);

    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(resize_w, dest_height), 0, 0, cv::INTER_LINEAR);

    cv::Mat float_resized;
    resized_img.convertTo(float_resized, CV_32FC3);

    // Normalize: (pixel - mean) / scale
    float_resized = (float_resized - cv::Scalar(NORM_MEAN, NORM_MEAN, NORM_MEAN)) / NORM_SCALE;

    // Calculate background padding value (maps to exactly 0 after quantization)
    float pad_value = -NORM_MEAN / NORM_SCALE;

    cv::Mat pre_image(dest_height, dest_width, CV_32FC3, cv::Scalar(pad_value, pad_value, pad_value));
    cv::Rect roi_rect(0, 0, resize_w, dest_height);
    float_resized.copyTo(pre_image(roi_rect));

    return pre_image;
}

// std::vector<int16_t> quantize_input_rec(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
// {
//     std::vector<int16_t> quantized_data;

//     if (float_img.empty() || float_img.type() != CV_32FC3)
//     {
//         std::cerr << "quantize_rec_tensor_int16: Invalid input image" << std::endl;
//         return quantized_data;
//     }

//     int total_elements = float_img.total() * float_img.channels();
//     quantized_data.resize(total_elements);

//     const float *src_ptr = float_img.ptr<float>();
//     float scale = attr.scale;
//     int32_t zp = attr.zp;

//     for (int i = 0; i < total_elements; ++i)
//     {
//         float val = std::round(src_ptr[i] / scale) + zp;
//         quantized_data[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, val)));
//     }

//     return quantized_data;
// }

std::vector<uint8_t> prepare_input_tensor(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;

    if (float_img.empty() || float_img.type() != CV_32FC3)
    {
        std::cerr << "prepare_input_tensor: Invalid input image" << std::endl;
        return tensor_data;
    }

    int total_elements = float_img.total() * float_img.channels();
    const float *src_ptr = float_img.ptr<float>();

    if (attr.type == AMLNN_TENSOR_FLOAT32)
    {
        tensor_data.resize(total_elements * sizeof(float));
        std::memcpy(tensor_data.data(), float_img.data, tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_FLOAT16)
    {
        cv::Mat fp16_img;
        float_img.convertTo(fp16_img, CV_16FC3);
        cv::Mat flat_img = fp16_img.isContinuous() ? fp16_img : fp16_img.clone();

        tensor_data.resize(total_elements * sizeof(uint16_t));
        std::memcpy(tensor_data.data(), flat_img.data, tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(total_elements * sizeof(int16_t));
        int16_t *dst_ptr = reinterpret_cast<int16_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
    }
    else
    {
        std::cerr << "prepare_input_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

std::string postprocess_rec(float *out_data, const std::vector<int> &out_shape, const std::vector<std::string> &char_dict)
{
    std::string result = "";
    if (out_data == nullptr || out_shape.size() < 2)
        return result;

    int seq_len = out_shape[out_shape.size() - 2];
    int num_classes = out_shape[out_shape.size() - 1];

    int blank_idx = 0;
    int pre_argmax_idx = -1;
    float total_score = 0.0f;
    int valid_char_count = 0;

    for (int t = 0; t < seq_len; ++t)
    {
        float raw_max_score = -1.0f;
        int argmax_idx = -1;

        // 1. Find the raw max score
        for (int c = 0; c < num_classes; ++c)
        {
            float val = out_data[t * num_classes + c];
            if (val > raw_max_score)
            {
                raw_max_score = val;
                argmax_idx = c;
            }
        }

        // 3. CTC Decoding logic
        if (argmax_idx != blank_idx && argmax_idx != pre_argmax_idx)
        {
            int char_idx = argmax_idx - 1;
            if (char_idx >= 0 && char_idx < char_dict.size())
            {
                result += char_dict[char_idx];
                valid_char_count++;
            }
        }
        pre_argmax_idx = argmax_idx;
    }

    return result;
}

cv::Mat draw_ocr_results(cv::Mat image, const std::vector<Object> &results)
{
    for (const auto &obj : results)
    {
        // 1. Draw Polygon Box
        cv::polylines(image, obj.box, true, cv::Scalar(0, 255, 0), 2);

        if (!obj.text.empty())
        {
            std::string label = obj.text + " (" + std::to_string(obj.rec_score).substr(0, 4) + ")";

            // Text properties
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.6;
            int thickness = 1;
            int baseline = 0;

            // 2. Calculate exact size of the text
            cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

            // 3. Calculate positioning
            cv::Point text_pos = obj.box[0];
            // Prevent text from going off the top of the screen
            text_pos.y = std::max(text_size.height, text_pos.y - 5);

            // 4. Calculate Background Rectangle Coordinates
            cv::Point bg_top_left(text_pos.x, text_pos.y - text_size.height - 2); // -2 for slight padding
            cv::Point bg_bottom_right(text_pos.x + text_size.width, text_pos.y + baseline + 2);

            // 5. Draw Solid Background (Green)
            cv::rectangle(image, bg_top_left, bg_bottom_right, cv::Scalar(0, 255, 0), cv::FILLED);

            // 6. Draw Text on top (Black text is easier to read on green)
            cv::putText(image, label, text_pos, font_face, font_scale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
        }
    }
    return image;
}
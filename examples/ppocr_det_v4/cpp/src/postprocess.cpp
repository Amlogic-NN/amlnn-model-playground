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
#include <limits>
#include <cstdint>
#include <cmath>

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

std::tuple<cv::Mat, float> preprocess(const cv::Mat &image, const int width, const int height)
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

std::vector<int8_t> quantize_input(const cv::Mat &float_img, const amlnn_tensor_attr &attr)
{
    std::vector<int8_t> quantized_data(float_img.total() * float_img.channels());
    const float *src = float_img.ptr<float>();

    for (size_t i = 0; i < quantized_data.size(); ++i)
    {
        float q_val = std::round(src[i] / attr.scale) + attr.zp;
        quantized_data[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, q_val)));
    }
    return quantized_data;
}

std::vector<Object> postprocess(float *out, const std::vector<int> &shape, const cv::Mat &image, float box_score_thresh, float box_thresh, float scale)
{
    if (out == NULL)
        return {};

    int H = MODEL_INPUT_HEIGHT;
    int W = MODEL_INPUT_WIDTH;

    cv::Mat pred_map(H, W, CV_32FC1, out);

    cv::Mat bit_map;
    bit_map = pred_map > box_thresh;
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(bit_map, bit_map, dila_ele, cv::Point(-1, -1), 1);

    return find_box(pred_map, bit_map, box_score_thresh, 1.5f, image, scale);
}

std::vector<Object> find_box(const cv::Mat pred_map, const cv::Mat &bit_map,
                             const float box_score_thresh, const float unclip_ratio,
                             const cv::Mat &image, float scale)
{

    std::vector<Object> res_boxes;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bit_map, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int num_coutours = contours.size() >= MAX_CANDIDATES ? MAX_CANDIDATES : contours.size();

    for (int i = 0; i < num_coutours; i++)
    {
        if (contours[i].size() <= 2)
            continue;

        float min_side_len;
        float perimeter;
        Object text_box;

        std::vector<cv::Point> min_box = get_min_boxes(contours[i], min_side_len, perimeter);

        if (min_side_len < MIN_SIZE)
            continue;

        // score
        float score = get_box_score_fast(pred_map, contours[i]);

        if (score < box_score_thresh)
            continue;

        //--- use clipper
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
        box[i].x = box[i].x - min_x;
        box[i].y = box[i].y - min_y;
    }

    std::vector<std::vector<cv::Point>> mask_box;
    mask_box.push_back(box);

    cv::Mat mask_mat(max_y - min_y + 1, max_x - min_x + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(mask_mat, mask_box, cv::Scalar(1, 1, 1), 1);

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
        ClipperLib::Path temp_poly = polys[i];
        for (int j = 0; j < temp_poly.size(); ++j)
        {
            out_box.emplace_back(temp_poly[j].X, temp_poly[j].Y);
        }
    }

    return out_box;
}

bool cv_point_compare(const cv::Point &a, const cv::Point &b)
{
    return a.x < b.x;
}

cv::Mat draw_objects(cv::Mat image, const std::vector<Object> &results)
{
    for (int i = 0; i < results.size(); i++)
    {
        cv::polylines(image, results[i].box, true, cv::Scalar(0, 0, 255), 2);
    }
    return image;
}
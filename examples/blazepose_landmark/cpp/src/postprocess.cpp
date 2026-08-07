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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

const int HEATMAP_SEARCH_RADIUS = 9;
const float HEATMAP_MIN_CONFIDENCE = 0.5f;
const int MODEL_SIZE = 256;
const float ROI_SCALE = 2.5f;
const float PI = 3.14159265358979323846f;

const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {1, 2}, {2, 3}, {3, 7}, {0, 4}, {4, 5}, {5, 6}, {6, 8},
    {9, 10}, {11, 12}, {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21},
    {17, 19}, {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22}, {18, 20},
    {11, 23}, {12, 24}, {23, 24}, {23, 25}, {24, 26}, {25, 27}, {26, 28},
    {27, 29}, {28, 30}, {29, 31}, {30, 32}, {27, 31}, {28, 32}
};

static float sigmoid(float value)
{
    value = std::max(-100.0f, std::min(value, 100.0f));
    return 1.0f / (1.0f + std::exp(-value));
}

static cv::Point2f roi_to_image_point(float x, float y, const Roi &roi)
{
    float local_x = (x / MODEL_SIZE - 0.5f) * roi.size;
    float local_y = (y / MODEL_SIZE - 0.5f) * roi.size;
    float cosine = std::cos(roi.rotation);
    float sine = std::sin(roi.rotation);

    return {roi.center_x + cosine * local_x - sine * local_y,
            roi.center_y + sine * local_x + cosine * local_y};
}

static int element_count(const std::vector<int> &shape)
{
    int count = 1;

    for (int dimension : shape)
        count *= dimension;

    return count;
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

std::vector<Detection> load_detections(const std::string &path)
{
    std::ifstream file(path);
    std::vector<Detection> detections;

    if (!file.is_open())
        return detections;

    while (true)
    {
        Detection detection;

        for (float &value : detection.coords)
        {
            if (!(file >> value))
                return detections;
        }

        if (!(file >> detection.score))
            return detections;

        detections.push_back(detection);
    }
}

Roi detection_to_roi(const Detection &detection, int image_width, int image_height)
{
    float center_x = detection.coords[4] * image_width;
    float center_y = detection.coords[5] * image_height;
    float end_x = detection.coords[6] * image_width;
    float end_y = detection.coords[7] * image_height;
    float radius = std::hypot(end_x - center_x, end_y - center_y);

    if (radius < 1.0f)
    {
        float box_width = (detection.coords[3] - detection.coords[1]) * image_width;
        float box_height = (detection.coords[2] - detection.coords[0]) * image_height;
        center_x = (detection.coords[1] + detection.coords[3]) * image_width / 2.0f;
        center_y = (detection.coords[0] + detection.coords[2]) * image_height / 2.0f;
        radius = std::max(box_width, box_height) / 2.0f;
    }

    float rotation = PI / 2.0f - std::atan2(-(end_y - center_y), end_x - center_x);
    return {center_x, center_y, ROI_SCALE * radius, rotation};
}

cv::Mat preprocess(const cv::Mat &image, const Roi &roi, std::tuple<int, int> new_shape)
{
    int input_height = std::get<0>(new_shape);
    int input_width = std::get<1>(new_shape);

    cv::Point2f source[3] = {
        roi_to_image_point(0.0f, 0.0f, roi),
        roi_to_image_point(input_width - 1.0f, 0.0f, roi),
        roi_to_image_point(0.0f, input_height - 1.0f, roi)
    };

    cv::Point2f destination[3] = {
        {0.0f, 0.0f},
        {static_cast<float>(input_width - 1), 0.0f},
        {0.0f, static_cast<float>(input_height - 1)}
    };

    cv::Mat transform = cv::getAffineTransform(source, destination);
    cv::Mat crop;

    cv::warpAffine(image, crop, transform, cv::Size(input_width, input_height),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    crop.convertTo(crop, CV_32FC3, 1.0 / 255.0);

    return crop;
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

static void refine_landmark_from_heatmap(const float *heatmap,
                                         int heatmap_height, int heatmap_width, int heatmap_channels,
                                         int landmark_index, float raw_x, float raw_y,
                                         int search_radius, float min_confidence,
                                         float &refined_x, float &refined_y)
{
    refined_x = raw_x;
    refined_y = raw_y;

    if (landmark_index < 0 || landmark_index >= heatmap_channels)
        return;

    int center_col = static_cast<int>(raw_x / MODEL_SIZE * heatmap_width);
    int center_row = static_cast<int>(raw_y / MODEL_SIZE * heatmap_height);

    if (center_col < 0 || center_col >= heatmap_width || center_row < 0 || center_row >= heatmap_height)
        return;

    int begin_col = std::max(0, center_col - search_radius);
    int end_col = std::min(heatmap_width, center_col + search_radius + 1);
    int begin_row = std::max(0, center_row - search_radius);
    int end_row = std::min(heatmap_height, center_row + search_radius + 1);

    float confidence_sum = 0.0f;
    float max_confidence = 0.0f;
    float weighted_col = 0.0f;
    float weighted_row = 0.0f;

    for (int row = begin_row; row < end_row; ++row)
    {
        for (int col = begin_col; col < end_col; ++col)
        {
            int index = (row * heatmap_width + col) * heatmap_channels + landmark_index;
            float confidence = sigmoid(heatmap[index]);

            confidence_sum += confidence;
            max_confidence = std::max(max_confidence, confidence);
            weighted_col += col * confidence;
            weighted_row += row * confidence;
        }
    }

    if (max_confidence < min_confidence || confidence_sum <= 0.0f)
        return;

    refined_x = weighted_col / (heatmap_width * confidence_sum) * MODEL_SIZE;
    refined_y = weighted_row / (heatmap_height * confidence_sum) * MODEL_SIZE;
}

bool postprocess(const std::vector<float *> &out_ptrs,
                 const std::vector<std::vector<int>> &out_shapes,
                 const Roi &roi, int image_width, int image_height,
                 float presence_threshold, PoseResult &result)
{
    result.score = out_ptrs[1][0];

    if (result.score < presence_threshold)
        return false;

    const std::vector<int> &heatmap_shape = out_shapes[3];

    if (heatmap_shape.size() != 3)
    {
        std::cerr << "Unexpected heatmap shape" << std::endl;
        return false;
    }

    int heatmap_height = heatmap_shape[0];
    int heatmap_width = heatmap_shape[1];
    int heatmap_channels = heatmap_shape[2];

    if (heatmap_channels < NUM_POSE_LANDMARKS)
    {
        std::cerr << "Heatmap has fewer channels than pose landmarks" << std::endl;
        return false;
    }

    const float *heatmap = out_ptrs[3];
    float cosine = std::cos(roi.rotation);
    float sine = std::sin(roi.rotation);

    for (int i = 0; i < NUM_POSE_LANDMARKS; ++i)
    {
        const float *raw = out_ptrs[0] + i * 5;
        const float *world = out_ptrs[4] + i * 3;

        float refined_x;
        float refined_y;

        refine_landmark_from_heatmap(
            heatmap,
            heatmap_height,
            heatmap_width,
            heatmap_channels,
            i,
            raw[0],
            raw[1],
            HEATMAP_SEARCH_RADIUS,
            HEATMAP_MIN_CONFIDENCE,
            refined_x,
            refined_y
        );

        cv::Point2f point = roi_to_image_point(refined_x, refined_y, roi);

        Landmark &landmark = result.landmarks[i];
        landmark.x = point.x / image_width;
        landmark.y = point.y / image_height;
        landmark.z = raw[2] * roi.size / (MODEL_SIZE * image_width);
        landmark.visibility = sigmoid(raw[3]);
        landmark.presence = sigmoid(raw[4]);
        landmark.world.x = cosine * world[0] - sine * world[1];
        landmark.world.y = sine * world[0] + cosine * world[1];
        landmark.world.z = world[2];
    }

    return true;
}

bool save_landmarks(const std::string &path, const std::vector<PoseResult> &results)
{
    std::ofstream file(path);

    if (!file.is_open())
        return false;

    file << std::fixed << std::setprecision(8);

    for (size_t pose_index = 0; pose_index < results.size(); ++pose_index)
    {
        for (int landmark_index = 0; landmark_index < NUM_POSE_LANDMARKS; ++landmark_index)
        {
            const Landmark &landmark = results[pose_index].landmarks[landmark_index];

            file << pose_index << ' ' << landmark_index << ' '
                 << landmark.x << ' ' << landmark.y << ' ' << landmark.z << ' '
                 << landmark.visibility << ' ' << landmark.presence << ' '
                 << landmark.world.x << ' ' << landmark.world.y << ' '
                 << landmark.world.z << '\n';
        }
    }

    return true;
}

cv::Mat draw_detections(const cv::Mat &image, const std::vector<PoseResult> &results,
                        float visibility_threshold)
{
    cv::Mat result_image = image.clone();

    for (const auto &result : results)
    {
        for (const auto &[a, b] : SKELETON)
        {
            const Landmark &landmark_a = result.landmarks[a];
            const Landmark &landmark_b = result.landmarks[b];

            if (landmark_a.visibility < visibility_threshold ||
                landmark_b.visibility < visibility_threshold)
                continue;

            cv::Point point_a(static_cast<int>(landmark_a.x * image.cols),
                              static_cast<int>(landmark_a.y * image.rows));

            cv::Point point_b(static_cast<int>(landmark_b.x * image.cols),
                              static_cast<int>(landmark_b.y * image.rows));

            cv::line(result_image, point_a, point_b, cv::Scalar(0, 255, 0), 2);
        }

        for (const auto &landmark : result.landmarks)
        {
            if (landmark.visibility < visibility_threshold)
                continue;

            int x = static_cast<int>(landmark.x * image.cols);
            int y = static_cast<int>(landmark.y * image.rows);

            if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
                cv::circle(result_image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    return result_image;
}
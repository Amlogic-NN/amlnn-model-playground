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
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>

const cv::Scalar IMAGENET_MEAN(123.675f, 116.280f, 103.530f);
const cv::Scalar IMAGENET_STD(58.395f, 57.120f, 57.375f);

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

cv::Mat load_image(const std::string &path, int input_height, int input_width)
{
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos)
        return {};

    std::string extension = path.substr(dot_pos);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

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

        file >> std::ws;
        if (!file.eof())
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
    if (tensor_type == AMLNN_TENSOR_INT8 || tensor_type == AMLNN_TENSOR_UINT8)
        return sizeof(uint8_t);

    return 0;
}

std::vector<uint8_t> load_direct_input_tensor(const std::string &path, const amlnn_tensor_attr &attr)
{
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos)
        return {};

    std::string extension = path.substr(dot_pos);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    size_t element_size = get_tensor_type_size(attr.type);
    if (element_size == 0)
    {
        std::cerr << "Direct .bin/.qtxt input only supports INT8 or UINT8 model inputs. Tensor type: "
                  << attr.type << std::endl;
        return {};
    }

    size_t expected_elements = static_cast<size_t>(attr.n_elems);
    size_t expected_size = expected_elements * element_size;

    if (extension == ".bin")
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cerr << "Failed to open BIN input: " << path << std::endl;
            return {};
        }

        std::streamsize file_size = file.tellg();
        if (file_size < 0 || static_cast<size_t>(file_size) != expected_size)
        {
            std::cerr << "Invalid BIN input size: expected " << expected_size
                      << " bytes, got " << file_size << ": " << path << std::endl;
            return {};
        }

        std::vector<uint8_t> data(expected_size);
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(expected_size));

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

        std::vector<uint8_t> data(expected_size);

        if (attr.type == AMLNN_TENSOR_INT8)
        {
            int8_t *dst = reinterpret_cast<int8_t *>(data.data());

            for (size_t i = 0; i < expected_elements; ++i)
            {
                int value;
                if (!(file >> value))
                {
                    std::cerr << "Invalid QTXT input data size: expected "
                              << expected_elements << " values: " << path << std::endl;
                    return {};
                }

                if (value < -128 || value > 127)
                {
                    std::cerr << "QTXT int8 value outside [-128, 127]: " << path << std::endl;
                    return {};
                }

                dst[i] = static_cast<int8_t>(value);
            }
        }
        else
        {
            uint8_t *dst = data.data();

            for (size_t i = 0; i < expected_elements; ++i)
            {
                int value;
                if (!(file >> value))
                {
                    std::cerr << "Invalid QTXT input data size: expected "
                              << expected_elements << " values: " << path << std::endl;
                    return {};
                }

                if (value < 0 || value > 255)
                {
                    std::cerr << "QTXT uint8 value outside [0, 255]: " << path << std::endl;
                    return {};
                }

                dst[i] = static_cast<uint8_t>(value);
            }
        }

        file >> std::ws;
        if (!file.eof())
        {
            std::cerr << "QTXT input contains unexpected extra data: " << path << std::endl;
            return {};
        }

        return data;
    }

    return {};
}

cv::Mat preprocess(cv::Mat img, std::tuple<int, int> new_shape)
{
    if (img.empty())
    {
        LOGE("Preprocess received empty image");
        return {};
    }

    cv::Mat img_rgb;
    if (img.channels() == 4)
        cv::cvtColor(img, img_rgb, cv::COLOR_RGBA2RGB);
    else if (img.channels() == 3)
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    else
        img_rgb = img.clone();

    int target_h = std::get<0>(new_shape);
    int target_w = std::get<1>(new_shape);

    cv::Mat resized_img;
    if (img_rgb.rows == target_h && img_rgb.cols == target_w)
        resized_img = img_rgb;
    else
        cv::resize(img_rgb, resized_img, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat img_float;
    resized_img.convertTo(img_float, CV_32FC3);
    cv::subtract(img_float, IMAGENET_MEAN, img_float);
    cv::divide(img_float, IMAGENET_STD, img_float);

    return img_float;
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

static void softmax(float *data, int size)
{
    float max_val = data[0];
    for (int i = 1; i < size; ++i)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    for (int i = 0; i < size; ++i)
    {
        data[i] /= sum;
    }
}

void postprocess_topk(float *buf, int size, const std::vector<std::string> &labels, int k)
{
    softmax(buf, size);
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort to get Top-K
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b)
                      { return buf[a] > buf[b]; });

    std::cout << "\n    Top-" << k << " Results:" << std::endl;
    for (int i = 0; i < k; ++i)
    {
        int idx = indices[i];
        std::string name = (idx < (int)labels.size()) ? labels[idx] : "Unknown(" + std::to_string(idx) + ")";
        printf("      %d. %-20s  prob=%.6f\n", i + 1, name.c_str(), buf[idx]);
    }
}

std::vector<std::string> load_labels(const std::string &path)
{
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "Warning: Could not open label file: " << path << std::endl;
        return labels;
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (!line.empty())
            labels.push_back(line);
    }
    return labels;
}
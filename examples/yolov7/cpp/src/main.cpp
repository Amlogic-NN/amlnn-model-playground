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

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <left_image> <right_image>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::string left_path = argv[2];
    std::string right_path = argv[3];
    std::cout << "CREStereo Demo" << std::endl;
    fs::create_directory("crestereo_result");

    // Initialize the network and query its input/output tensor metadata once.
    void *context = nullptr;
    int ret = init_network(model_path, context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_input != 4)
    {
        std::cerr << "Expected 4 CREStereo inputs, but model has " << io_num.n_input << " inputs." << std::endl;
        uninit_network(context);
        return -1;
    }
    if (io_num.n_output != 1)
    {
        std::cerr << "Expected 1 CREStereo output, but model has " << io_num.n_output << " outputs." << std::endl;
        uninit_network(context);
        return -1;
    }

    // Physical input order: init_left, init_right, next_left, next_right.
    std::vector<amlnn_tensor_attr> input_attrs;
    std::vector<std::vector<int>> input_shapes;
    for (int i = 0; i < io_num.n_input; ++i)
    {
        input_attrs.push_back(query_input_attr(context, i));
        input_shapes.push_back(get_tensor_shape(input_attrs[i]));
    }

    if (input_shapes[0] != input_shapes[1])
    {
        std::cerr << "Init left/right input shapes differ." << std::endl;
        uninit_network(context);
        return -1;
    }
    if (input_shapes[2] != input_shapes[3])
    {
        std::cerr << "Next left/right input shapes differ." << std::endl;
        uninit_network(context);
        return -1;
    }
    if (input_shapes[0].size() != 3 || input_shapes[0][2] != 3)
    {
        std::cerr << "Invalid init input shape." << std::endl;
        uninit_network(context);
        return -1;
    }
    if (input_shapes[2].size() != 3 || input_shapes[2][2] != 3)
    {
        std::cerr << "Invalid next input shape." << std::endl;
        uninit_network(context);
        return -1;
    }

    std::cout << "Init input shape: [";
    for (size_t i = 0; i < input_shapes[0].size(); ++i)
    {
        std::cout << input_shapes[0][i];
        if (i + 1 < input_shapes[0].size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Next input shape: [";
    for (size_t i = 0; i < input_shapes[2].size(); ++i)
    {
        std::cout << input_shapes[2][i];
        if (i + 1 < input_shapes[2].size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    amlnn_tensor_attr output_attr = query_output_attr(context, 0);
    std::vector<int> output_shape = get_tensor_shape(output_attr);
    if (output_shape.size() != 3 || output_shape[2] != 2)
    {
        std::cerr << "Expected CREStereo output shape [H, W, 2]." << std::endl;
        uninit_network(context);
        return -1;
    }

    cv::Mat left_img = cv::imread(left_path);
    if (left_img.empty())
    {
        std::cerr << "Can't read left image: " << left_path << std::endl;
        uninit_network(context);
        return -1;
    }

    cv::Mat right_img = cv::imread(right_path);
    if (right_img.empty())
    {
        std::cerr << "Can't read right image: " << right_path << std::endl;
        uninit_network(context);
        return -1;
    }

    if (left_img.rows != right_img.rows || left_img.cols != right_img.cols)
    {
        std::cerr << "Left/right image sizes differ." << std::endl;
        uninit_network(context);
        return -1;
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "Left image: \"" << fs::path(left_path).filename().string() << "\"" << std::endl;
    std::cout << "Right image: \"" << fs::path(right_path).filename().string() << "\"" << std::endl;
    std::cout << "============================================================" << std::endl;

    std::tuple<int, int> init_shape = std::make_tuple(input_shapes[0][0], input_shapes[0][1]);
    std::tuple<int, int> next_shape = std::make_tuple(input_shapes[2][0], input_shapes[2][1]);

    cv::Mat init_left = preprocess(left_img, init_shape);
    cv::Mat init_right = preprocess(right_img, init_shape);
    cv::Mat next_left = preprocess(left_img, next_shape);
    cv::Mat next_right = preprocess(right_img, next_shape);

    std::vector<uint8_t> init_left_data = prepare_input_tensor(init_left, input_attrs[0]);
    std::vector<uint8_t> init_right_data = prepare_input_tensor(init_right, input_attrs[1]);
    std::vector<uint8_t> next_left_data = prepare_input_tensor(next_left, input_attrs[2]);
    std::vector<uint8_t> next_right_data = prepare_input_tensor(next_right, input_attrs[3]);
    if (init_left_data.empty() || init_right_data.empty() || next_left_data.empty() || next_right_data.empty())
    {
        std::cerr << "Failed to prepare input tensors." << std::endl;
        uninit_network(context);
        return -1;
    }

    std::vector<void *> input_ptrs = {
        init_left_data.data(), init_right_data.data(),
        next_left_data.data(), next_right_data.data()};
    std::vector<size_t> input_sizes = {
        init_left_data.size(), init_right_data.size(),
        next_left_data.size(), next_right_data.size()};
    std::vector<amlnn_output> out_data(io_num.n_output);

    auto start_time = std::chrono::high_resolution_clock::now();
    if (!run_multi_input_network(context, input_ptrs, input_sizes, out_data))
    {
        std::cerr << "Failed to run network" << std::endl;
        uninit_network(context);
        return -1;
    }

    if (out_data.empty() || out_data[0].buf == nullptr)
    {
        std::cerr << "Network returned no output." << std::endl;
        uninit_network(context);
        return -1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
    std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

    // Runtime output buffers are already dequantized and exposed as float arrays.
    float *output_ptr = static_cast<float *>(out_data[0].buf);
    cv::Mat disparity = postprocess(output_ptr, output_shape, left_img.rows, left_img.cols);
    if (disparity.empty())
    {
        std::cerr << "Failed to postprocess disparity output." << std::endl;
        uninit_network(context);
        return -1;
    }

    cv::Mat result_img = colorize_disparity(disparity);
    std::string out_path = "crestereo_result/" + fs::path(left_path).stem().string() + "_result.jpg";
    if (!cv::imwrite(out_path, result_img))
    {
        std::cerr << "Failed to save result image: " << out_path << std::endl;
        uninit_network(context);
        return -1;
    }

    std::cout << "Result saved to: " << out_path << std::endl;
    std::cout << "============================================================" << std::endl
              << std::endl;

    uninit_network(context);
    return 0;
}
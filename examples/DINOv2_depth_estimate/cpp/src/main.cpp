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
        std::cout << "Usage: " << argv[0] << " <backbone.adla> <depth.adla> <image_dir> [min_depth] [max_depth]\n";
        return 0;
    }

    std::string backbone_model_path = argv[1];
    std::string depth_model_path = argv[2];
    std::string image_dir = argv[3];
    float min_depth = argc > 4 ? std::stof(argv[4]) : 0.001f;
    float max_depth = argc > 5 ? std::stof(argv[5]) : 10.0f;

    std::cout << "DINOv2 NYU DPT Depth Demo" << std::endl;
    fs::create_directory("dinov2_dd_result");

    void *backbone_context = nullptr;
    void *depth_context = nullptr;

    int ret = init_network(backbone_model_path, backbone_context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize backbone network. Error: " << ret << std::endl;
        return -1;
    }

    ret = init_network(depth_model_path, depth_context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize depth network. Error: " << ret << std::endl;
        uninit_network(backbone_context);
        return -1;
    }

    amlnn_input_output_num backbone_io_num;
    amlnn_query(backbone_context, AMLNN_QUERY_IN_OUT_NUM, &backbone_io_num, sizeof(backbone_io_num));

    amlnn_input_output_num depth_io_num;
    amlnn_query(depth_context, AMLNN_QUERY_IN_OUT_NUM, &depth_io_num, sizeof(depth_io_num));

    amlnn_tensor_attr backbone_input_attr = query_input_attr(backbone_context, 0);
    std::vector<int> backbone_input_shape = get_tensor_shape(backbone_input_attr);

    std::cout << "Backbone input shape: [";
    for (size_t i = 0; i < backbone_input_shape.size(); ++i)
    {
        std::cout << backbone_input_shape[i];
        if (i + 1 < backbone_input_shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    int input_height = backbone_input_shape[0];
    int input_width = backbone_input_shape[1];

    std::vector<amlnn_tensor_attr> backbone_output_attrs;
    std::vector<std::vector<int>> backbone_output_shapes;
    for (int i = 0; i < backbone_io_num.n_output; ++i)
    {
        backbone_output_attrs.push_back(query_output_attr(backbone_context, i));
        backbone_output_shapes.push_back(get_tensor_shape(backbone_output_attrs[i]));

        std::cout << "Backbone output " << i << " shape: [";
        for (size_t j = 0; j < backbone_output_shapes[i].size(); ++j)
        {
            std::cout << backbone_output_shapes[i][j];
            if (j + 1 < backbone_output_shapes[i].size())
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    amlnn_tensor_attr depth_input_attr = query_input_attr(depth_context, 0);
    amlnn_tensor_attr depth_output_attr = query_output_attr(depth_context, 0);
    std::vector<int> depth_input_shape = get_tensor_shape(depth_input_attr);
    std::vector<int> depth_output_shape = get_tensor_shape(depth_output_attr);

    std::cout << "Depth input shape: [";
    for (size_t i = 0; i < depth_input_shape.size(); ++i)
    {
        std::cout << depth_input_shape[i];
        if (i + 1 < depth_input_shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Depth output shape: [";
    for (size_t i = 0; i < depth_output_shape.size(); ++i)
    {
        std::cout << depth_output_shape[i];
        if (i + 1 < depth_output_shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::vector<amlnn_output> backbone_outData(backbone_io_num.n_output);
    std::vector<amlnn_output> depth_outData(depth_io_num.n_output);

    for (auto &it : fs::directory_iterator(image_dir))
    {
        if (!it.is_regular_file())
            continue;

        cv::Mat img = cv::imread(it.path().string());
        if (img.empty())
            continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        cv::Mat preprocessed = preprocess(img, std::make_tuple(input_height, input_width));
        std::vector<uint8_t> prepared_backbone_input = prepare_input_tensor(preprocessed, backbone_input_attr);

        if (prepared_backbone_input.empty())
        {
            std::cerr << "Failed to prepare backbone input tensor." << std::endl;
            continue;
        }

        auto backbone_start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(backbone_context, prepared_backbone_input.data(), prepared_backbone_input.size(), backbone_outData))
        {
            std::cerr << "Failed to run backbone network" << std::endl;
            return -1;
        }

        auto backbone_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> backbone_inference_time = backbone_end_time - backbone_start_time;
        std::cout << "Backbone inference time: " << backbone_inference_time.count() << " ms" << std::endl;

        if (backbone_outData.empty())
            return -1;

        std::vector<float> concat_features = concatenate_backbone_outputs(
            backbone_outData,
            backbone_output_shapes,
            depth_input_attr.n_elems);

        if (concat_features.empty())
        {
            std::cerr << "Failed to concatenate backbone outputs." << std::endl;
            continue;
        }

        std::vector<uint8_t> prepared_depth_input = prepare_feature_tensor(concat_features, depth_input_attr);

        if (prepared_depth_input.empty())
        {
            std::cerr << "Failed to prepare depth input tensor." << std::endl;
            continue;
        }

        auto depth_start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(depth_context, prepared_depth_input.data(), prepared_depth_input.size(), depth_outData))
        {
            std::cerr << "Failed to run depth network" << std::endl;
            return -1;
        }

        auto depth_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> depth_inference_time = depth_end_time - depth_start_time;
        std::cout << "Depth inference time: " << depth_inference_time.count() << " ms" << std::endl;

        if (depth_outData.empty())
            return -1;

        float *depth_output = reinterpret_cast<float *>(depth_outData[0].buf);
        cv::Mat depth_map = postprocess(
            depth_output,
            depth_output_attr.n_elems,
            depth_output_shape,
            img.size(),
            min_depth,
            max_depth);

        if (depth_map.empty())
        {
            std::cerr << "Failed to postprocess depth output." << std::endl;
            continue;
        }

        cv::Mat depth_color = colorize_depth(depth_map);
        std::string output_path = "dinov2_dd_result/" + it.path().stem().string() + "_depth.png";
        cv::imwrite(output_path, depth_color);

        double depth_min;
        double depth_max;
        cv::minMaxLoc(depth_map, &depth_min, &depth_max);
        cv::Scalar depth_mean = cv::mean(depth_map);

        std::cout << "Results:" << std::endl;
        std::cout << "  Depth shape: [" << depth_map.rows << ", " << depth_map.cols << "]" << std::endl;
        std::cout << "  Depth range: " << depth_min << " to " << depth_max << std::endl;
        std::cout << "  Mean depth: " << depth_mean[0] << std::endl;
        std::cout << "  Visualization: " << output_path << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    uninit_network(backbone_context);
    uninit_network(depth_context);

    return 0;
}
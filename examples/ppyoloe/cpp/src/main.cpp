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

const float SCORE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.2f;
const int REG_MAX = 16;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::cout << "PP-YOLOE Demo" << std::endl;
    fs::create_directory("ppyoloe_result");

    // Initialize the network and cache its tensor metadata.
    void *context = nullptr;
    int ret = init_network(model_path, context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_output != 6)
    {
        std::cerr << "Expected 6 PP-YOLOE outputs, but model has "
                  << io_num.n_output << " outputs." << std::endl;
        uninit_network(context);
        return -1;
    }

    amlnn_tensor_attr input_attr = query_input_attr(context, 0);
    std::vector<int> input_shape = get_tensor_shape(input_attr);
    if (input_shape.size() < 2)
    {
        std::cerr << "Invalid input shape." << std::endl;
        uninit_network(context);
        return -1;
    }

    int input_height = input_shape[0];
    int input_width = input_shape[1];
    std::vector<amlnn_tensor_attr> out_attrs;
    std::vector<std::vector<int>> out_shapes;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_attrs.push_back(query_output_attr(context, i));
        out_shapes.push_back(get_tensor_shape(out_attrs[i]));
    }

    std::vector<amlnn_output> out_data(io_num.n_output);

    // Process every readable image in the supplied directory.
    for (auto &it : fs::directory_iterator(argv[2]))
    {
        if (!it.is_regular_file())
            continue;

        cv::Mat img = cv::imread(it.path().string());
        if (img.empty())
            continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        auto [preprocessed, scale_x, scale_y] = preprocess(
            img, std::make_tuple(input_height, input_width)
        );
        std::vector<uint8_t> prepared_data = prepare_input_tensor(preprocessed, input_attr);
        if (prepared_data.empty())
        {
            std::cerr << "Failed to prepare input tensor." << std::endl;
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        if (!run_network(context, prepared_data.data(), prepared_data.size(), out_data))
        {
            std::cerr << "Failed to run network" << std::endl;
            uninit_network(context);
            return -1;
        }

        if (out_data.empty())
        {
            uninit_network(context);
            return -1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        // Runtime outputs have already been dequantized into float buffers.
        std::vector<float *> out_ptrs;
        for (int i = 0; i < io_num.n_output; ++i)
            out_ptrs.push_back(static_cast<float *>(out_data[i].buf));

        std::vector<Detection> detections = postprocess(
            out_ptrs, out_shapes, input_height, input_width, scale_x, scale_y,
            SCORE_THRESHOLD, NMS_THRESHOLD, REG_MAX
        );

        std::cout << "Detections: " << detections.size() << std::endl;
        cv::Mat result_img = draw_detections(img, detections);
        std::string out_path = "ppyoloe_result/" + it.path().filename().string();
        cv::imwrite(out_path, result_img);
        std::cout << "Result saved to: " << out_path << std::endl;
    }

    std::cout << "============================================================" << std::endl << std::endl;
    uninit_network(context);
    return 0;
}
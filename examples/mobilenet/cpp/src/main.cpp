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
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"
#include <filesystem>

const int MODEL_INPUT_WIDTH = 224;
const int MODEL_INPUT_HEIGHT = 224;
const int TOP_K = 5;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir> [labels.txt (optional)]\n";
        return 0;
    }

    std::string model_path = argv[1];

    std::string labels_path = "../input/labels.txt";

    if (argc >= 4)
    {
        labels_path = argv[3];
    }

    std::cout << "MobileNetV2 Demo" << std::endl;

    if (!fs::exists(labels_path))
    {
        std::cerr << "Error: Labels file not found at path: " << labels_path << std::endl;
        return -1;
    }

    fs::create_directory("MobileNetV2_result");

    std::vector<std::string> labels = load_labels(labels_path);
    // 1. Initialize Network
    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    // Query IO numbers to ensure we have exactly 1 output
    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_output != 1)
    {
        std::cerr << "Warning: Expected 1 outputs (boxes, scores), but model has "
                  << io_num.n_output << " outputs." << std::endl;
    }

    // Query Input Attribute for Scale and Zero Point
    amlnn_tensor_attr input_attr = query_input_attr(context, 0);

    std::vector<amlnn_output> outData(1);

    for (auto &it : fs::directory_iterator(argv[2]))
    {
        if (!it.is_regular_file())
            continue;

        // 2. Load Image
        cv::Mat img = cv::imread(it.path().string());
        if (img.empty())
            continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        // 3. Preprocess
        auto [preprocessed, scale, pad] = preprocess(img, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));
        cv::Mat quantized_img = quantize_input(preprocessed, input_attr);

        // 4. Set input, run inference, and Get Outputs
        size_t input_size = input_attr.n_elems * sizeof(int8_t);

        auto start_time = std::chrono::high_resolution_clock::now();
        if (!run_network(context, quantized_img.data, input_size, outData))
        {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }

        if (outData.empty())
        {
            return -1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        float *out_buffer = (float *)outData[0].buf;

        // 5. Postprocess: Get Top K
        int size = outData[0].size / sizeof(float);
        std::cout << "============================================================" << std::endl;
        postprocess_topk(out_buffer, size, labels, TOP_K);
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 6. Cleanup
    uninit_network(context);

    return 0;
}
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
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

const int MODEL_INPUT_WIDTH = 512;
const int MODEL_INPUT_HEIGHT = 512;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];

    std::cout << "DeepLabV3 Semantic Segmentation Demo" << std::endl;

    fs::create_directory("deeplab_result");

    // 1. Initialize Network
    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    // Query IO numbers
    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    // Query Input Attribute for Scale, Zero Point, and Type
    amlnn_tensor_attr input_attr = query_input_attr(context, 0);

    // Cache Output Shapes
    std::vector<std::vector<int>> out_shapes;
    for (int i = 0; i < io_num.n_output; i++)
    {
        amlnn_tensor_attr curr = query_output_attr(context, i);
        out_shapes.push_back(get_tensor_shape(curr));
    }

    std::vector<amlnn_output> outData(io_num.n_output);

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
        auto [float_img, scale, pad_left, pad_top, new_w, new_h] = preprocess(img, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

        // Quantize using your robust helper
        std::vector<uint8_t> quantized_buffer = prepare_input_tensor(float_img, input_attr);

        if (quantized_buffer.empty()) {
            std::cerr << "Failed to prepare input tensor." << std::endl;
            continue;
        }

        // 4. Run inference
        auto start_time = std::chrono::high_resolution_clock::now();

        size_t input_size = quantized_buffer.size();
        if (!run_network(context, quantized_buffer.data(), input_size, outData))
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

        // 5. Postprocess
        cv::Mat mask = postprocess(
            (float *)outData[0].buf, out_shapes[0],
            img.cols, img.rows, pad_left, pad_top, new_w, new_h
        );

        // 6. Draw and Save
        cv::Mat result_img = draw_segmentation(img, mask);
        std::string out_path = "deeplab_result/" + it.path().filename().string();
        cv::imwrite(out_path, result_img);
        std::cout << "Result saved to: " << out_path << std::endl;
    }

    std::cout << "============================================================" << std::endl << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}
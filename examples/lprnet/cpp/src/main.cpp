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
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include "nnsdk2.h"
#include "model_loader.h"
#include "postprocess.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <model.adla> <image_dir>\n", argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_dir = argv[2];

    std::cout << "LPRNet (Chinese) Demo" << std::endl;

    // Create result directory based on model name
    std::string model_stem = fs::path(model_path).stem().string();
    std::string result_dir = model_stem + "_result";
    fs::create_directory(result_dir);

    // 1. Initialize Network
    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    amlnn_tensor_attr input_attr = query_input_attr(context, 0);
    amlnn_tensor_attr output_attr = query_output_attr(context, 0);
    std::vector<int> out_shape = get_tensor_shape(output_attr);

    printf("LPR Input Type: %d, Scale: %.8f, ZP: %d\n", input_attr.type, input_attr.scale, input_attr.zp);

    std::vector<amlnn_output> outData(io_num.n_output);

    for (auto &it : fs::directory_iterator(image_dir))
    {
        if (!it.is_regular_file())
            continue;

        // 2. Load Image
        cv::Mat img = cv::imread(it.path().string());
        if (img.empty())
        {
            std::cerr << "Failed to load image from " << it.path().string() << std::endl;
            continue;
        }

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        // 3. Preprocess
        cv::Mat float_image = preprocess(img, LPR_MODEL_WIDTH, LPR_MODEL_HEIGHT);

        // Dynamic input formatter
        std::vector<uint8_t> prepared_data = prepare_input_tensor(float_image, input_attr);
        if (prepared_data.empty())
        {
            std::cerr << "Failed to prepare input tensor." << std::endl;
            continue;
        }

        // 4. Inference
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(context, prepared_data.data(), prepared_data.size(), outData))
        {
            std::cerr << "Failed to run network" << std::endl;
            uninit_network(context);
            return -1;
        }

        if (outData.empty() || outData[0].buf == nullptr)
        {
            std::cerr << "Invalid output data" << std::endl;
            continue;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        // 5. Postprocess (CTC Decoder)
        auto [text, score] = postprocess_lpr((float *)outData[0].buf, out_shape);

        // 6. Print and Save Result
        printf("    Recognized Plate: [%s]\n", text.c_str());

        cv::Mat result_img = draw_detections(img, text, score);
        std::string save_path = result_dir + "/" + it.path().filename().stem().string() + "_result.jpg";
        cv::imwrite(save_path, result_img);

        std::cout << "    Image saved to:  " << save_path << std::endl;
    }

    std::cout << "============================================================" << std::endl << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}
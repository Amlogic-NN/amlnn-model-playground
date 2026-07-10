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
    if (argc < 4)
    {
        printf("Usage: %s <model.adla> <image_dir> <dict_path>\n", argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string dict_path = argv[3];

    std::cout << "PPOCR Rec Demo (Robust Type Handler)" << std::endl;

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

    printf("REC Input Type: %d, Scale: %.8f, ZP: %d\n", input_attr.type, input_attr.scale, input_attr.zp);

    std::vector<amlnn_output> outData(io_num.n_output);

    std::vector<std::string> char_dict = load_dict(dict_path);
    if (char_dict.empty())
    {
        std::cerr << "Failed to retrieve dictionary " << ret << std::endl;
        return -1;
    }

    for (auto &it : fs::directory_iterator(argv[2]))
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
        cv::Mat float_image = preprocess(img, REC_MODEL_INPUT_WIDTH, REC_MODEL_INPUT_HEIGHT);

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
            return -1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        // 5. Postprocess
        std::string result = postprocess_rec((float *)outData[0].buf, out_shape, char_dict);

        // 6. Print output
        printf("[RESULT] Recognized Text: %s\n", result.c_str());
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}
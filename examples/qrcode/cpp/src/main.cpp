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
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

const int MODEL_INPUT_WIDTH = 320;
const int MODEL_INPUT_HEIGHT = 320;
const float CONF_THRESHOLD = 0.8f;
const float NMS_THRESHOLD = 0.5f;
const int PAD = 40;

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];

    std::cout << "QRCode Detection & Decoding Demo" << std::endl;

    std::string res_dir = "qrcode_result";
    fs::create_directory(res_dir);

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
    if (io_num.n_output != 1)
    {
        std::cerr << "Warning: Expected 1 output, but model has "
                  << io_num.n_output << " outputs." << std::endl;
    }

    // Query Input/Output Attributes
    amlnn_tensor_attr input_attr = query_input_attr(context, 0);
    amlnn_tensor_attr output_attr = query_output_attr(context, 0);
    std::vector<int> out_shape = get_tensor_shape(output_attr);

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
        auto [preprocessed, sx, sy] = preprocess(img, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

        // 4. Run inference
        auto start_time = std::chrono::high_resolution_clock::now();

        size_t input_size = input_attr.n_elems * sizeof(float32_t);
        if (!run_network(context, preprocessed.data, input_size, outData))
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
        std::cout << "    Inference time: " << inference_time.count() << " ms" << std::endl;

        // 5. Postprocess
        std::vector<Detection> raw_detections = postprocess(
            (float *)outData[0].buf, out_shape,
            sx, sy, img.cols, img.rows,
            CONF_THRESHOLD, NMS_THRESHOLD, PAD);

        // 6. Decode QR Codes
        std::vector<Detection> final_results = decode(img, raw_detections);

        if (final_results.empty())
        {
            std::cout << "    No objects detected" << std::endl;
        }
        else
        {
            std::cout << "    Detected " << final_results.size() << " objects:" << std::endl;
            for (size_t i = 0; i < final_results.size(); ++i)
            {
                const auto &r = final_results[i];
                std::cout << "      " << (i + 1) << ". score=" << std::fixed << std::setprecision(3) << r.score << "\n";
                std::cout << "         box=[" << r.x1 << ", " << r.y1 << ", " << r.x2 << ", " << r.y2 << "]\n";
                std::cout << "         text=" << r.text << "\n";
            }
        }

        // 7. Draw and Save
        cv::Mat result_img = draw_results(img, final_results);
        std::string save_path = res_dir + "/" + it.path().filename().string();
        cv::imwrite(save_path, result_img);
        std::cout << "    Result saved to: " << save_path << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 8. Cleanup
    uninit_network(context);

    return 0;
}
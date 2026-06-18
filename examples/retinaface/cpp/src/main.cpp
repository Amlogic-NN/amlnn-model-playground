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
#include <vector>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "nnsdk2.h"
#include "postprocess.h"
#include "model_loader.h"

namespace fs = std::filesystem;

const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.4f;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::string out_dir = "retinaface_result";

    std::cout << "RetinaFace Demo" << std::endl;
    fs::create_directory(out_dir);

    // 1. Initialize Network
    void *context = nullptr;
    int ret = init_network(model_path, context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network." << std::endl;
        return -1;
    }

    // Query IO numbers
    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_output != 3)
    {
        std::cerr << "Warning: Expected 3 outputs, but model has " << io_num.n_output << std::endl;
    }

    // Query Input Attributes
    amlnn_tensor_attr input_attr = query_input_attr(context, 0);
    size_t input_size = input_attr.n_elems * sizeof(int8_t);

    std::vector<amlnn_output> outData(io_num.n_output);
    int expected_priors = get_num_priors(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

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
        auto input_tuple = preprocess(img, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
        cv::Mat float_img = std::get<0>(input_tuple);
        cv::Mat quantized_img = quantize_input(float_img, input_attr.scale, input_attr.zp, input_attr.type);

        // 4. Run Inference
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(context, quantized_img.data, input_size, outData))
        {
            std::cerr << "Failed to run network" << std::endl;
            continue;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Inference time: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

        // 5. Locate Outputs & Determine Shape (NCHW vs NHWC handling)
        float *loc = nullptr, *conf = nullptr, *landm = nullptr;
        bool loc_planar = false, conf_planar = false, landm_planar = false;

        for (int j = 0; j < io_num.n_output; j++)
        {
            amlnn_tensor_attr attr = query_output_attr(context, j);
            int total_elems = outData[j].size / sizeof(float);
            int last_dim = attr.dims[attr.n_dims - 1];

            if (total_elems == expected_priors * 4)
            {
                loc = (float *)outData[j].buf;
                loc_planar = (last_dim != 4);
            }
            else if (total_elems == expected_priors * 2)
            {
                conf = (float *)outData[j].buf;
                conf_planar = (last_dim != 2);
            }
            else if (total_elems == expected_priors * 10)
            {
                landm = (float *)outData[j].buf;
                landm_planar = (last_dim != 10);
            }
        }

        if (!loc || !conf || !landm)
        {
            std::cerr << "Output parsing failed! Sizes did not match expected shapes." << std::endl;
            continue;
        }

        // 6. Postprocess wrapper
        std::vector<FaceDetection> detections = postprocess(
            loc, loc_planar, conf, conf_planar, landm, landm_planar,
            input_tuple, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, SCORE_THRESHOLD, NMS_THRESHOLD);

        std::cout << "    Detected " << detections.size() << " faces\n";

        // 7. Draw and Save
        cv::Mat result_img = draw_detections(img, detections);

        std::string save_path = out_dir + "/" + it.path().filename().string();
        cv::imwrite(save_path, result_img);
        std::cout << "    Result saved to: " << save_path << "\n";
    }

    std::cout << "============================================================" << std::endl;

    // 8. Cleanup
    uninit_network(context);
    return 0;
}
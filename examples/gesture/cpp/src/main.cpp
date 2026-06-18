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

const int MODEL_INPUT_SIZE = 640;
const float SCORE_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.3f;
namespace fs = std::filesystem;

extern const char *NAMES[19];

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <model_path> <image_dir> [--top1-only]\n", argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string out_dir = "gesture_result";

    bool top1_only = false;
    for (int i = 3; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--top1-only")
            top1_only = true;
    }

    std::cout << "Gesture Demo" << std::endl;
    fs::create_directory(out_dir);

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
    if (io_num.n_output != 3)
    {
        std::cerr << "Warning: Expected 3 outputs, but model has "
                  << io_num.n_output << " outputs." << std::endl;
    }

    // Query Input Attributes
    amlnn_tensor_attr input_attr = query_input_attr(context, 0);

    // Cache Output Shapes
    std::vector<std::vector<int>> out_shapes;
    for (int i = 0; i < io_num.n_output; i++)
    {
        out_shapes.push_back(get_tensor_shape(query_output_attr(context, i)));
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

        // 3. Preprocess & Quantize
        auto [preprocessed, orig_w, orig_h] = preprocess(img, MODEL_INPUT_SIZE);

        // 4. Run Inference
        size_t input_size = input_attr.n_elems * sizeof(float32_t);

        auto start_time = std::chrono::high_resolution_clock::now();
        if (!run_network(context, preprocessed.data, input_size, outData))
        {
            std::cerr << "Failed to run network" << std::endl;
            continue;
        }

        if (outData.empty())
        {
            return -1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        std::vector<float *> out_ptrs = {
            (float *)outData[0].buf,
            (float *)outData[1].buf,
            (float *)outData[2].buf};

        // 5. Postprocess
        std::vector<GestureDetection> detections = postprocess(
            out_ptrs, out_shapes,
            std::make_tuple(preprocessed, orig_w, orig_h),
            SCORE_THRESHOLD,
            NMS_THRESHOLD);

        // Apply --top1-only filter
        if (top1_only && !detections.empty())
        {
            detections.resize(1);
        }

        // 6. Console Logging
        if (detections.empty())
        {
            std::cout << "    No objects detected" << std::endl;
        }
        else
        {
            std::cout << "    Detected " << detections.size() << " objects:" << std::endl;
            for (size_t i = 0; i < detections.size(); ++i)
            {
                std::cout << "      " << (i + 1) << ". class=" << NAMES[detections[i].class_id] << "\n"
                          << "         score=" << std::fixed << std::setprecision(3) << detections[i].score << "\n"
                          << "         box=[" << static_cast<int>(detections[i].x1) << ", "
                          << static_cast<int>(detections[i].y1) << ", "
                          << static_cast<int>(detections[i].x2) << ", "
                          << static_cast<int>(detections[i].y2) << "]" << std::endl;
            }
        }

        // 7. Draw and Save Image
        cv::Mat result_img = draw_detections(img, detections);
        std::string out_path = out_dir + "/" + it.path().filename().string();
        cv::imwrite(out_path, result_img);
        std::cout << "\nResult saved to: " << out_path << std::endl
                  << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 8. Cleanup
    uninit_network(context);

    return 0;
}
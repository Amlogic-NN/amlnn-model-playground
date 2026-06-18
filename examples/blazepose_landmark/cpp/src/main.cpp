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
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

const int MODEL_INPUT_WIDTH = 256;
const int MODEL_INPUT_HEIGHT = 256;
const float SCORE_THRESHOLD = 0.5f;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir> <detections_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::string det_dir = argv[3];

    std::cout << "Blazepose Landmark Demo" << std::endl;

    fs::create_directory("blazepose_landmark_result");

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

    int landmarks_idx = -1;
    int heatmap_idx = -1;

    for (int i = 0; i < io_num.n_output; i++)
    {
        amlnn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        amlnn_query(context, AMLNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));

        if (attr.n_elems == 195)
        {
            // 39 keypoints * 5 dims (x, y, z, vis, presence)
            landmarks_idx = i;
        }
        else if (attr.n_elems == 159744)
        {
            // Actual Heatmap (64x64x39)
            heatmap_idx = i;
        }
    }

    if (landmarks_idx == -1) {
        std::cerr << "Error: Could not find landmark output tensor (size 195)!" << std::endl;
        return -1;
    }

    std::cout << "Mapped Landmarks to output index: " << landmarks_idx
              << ", Heatmap to output index: " << heatmap_idx << std::endl;

    // Query Input Attribute for Scale and Zero Point
    amlnn_tensor_attr input_attr = query_input_attr(context, 0);

    // Ensure API outputs Float32 directly
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

        // Load detection text file
        std::string filename = it.path().stem().string();
        std::string txt_path = det_dir + "/" + filename + ".txt";
        auto detections = load_detections(txt_path);

        if (detections.empty())
        {
            std::cout << "No detections found, skipping..." << std::endl;
            continue;
        }

        // 3. Preprocess
        auto [preprocessed, roi] = preprocess(img, detections, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));
        cv::Mat quantized_img = quantize_input(preprocessed, input_attr);

        // 4. Set input, run inference, and Get Outputs
        auto start_time = std::chrono::high_resolution_clock::now();

        size_t input_size = input_attr.n_elems * sizeof(int8_t);
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

        // 5. Postprocess (Direct Float Cast)
        std::vector<BlazePoseLandmark> landmarks = postprocess(
            (float *)outData[landmarks_idx].buf, // Dynamically found landmarks
            (float *)outData[heatmap_idx].buf,   // Dynamically found heatmap / presence
            roi);
        std::cout << "Landmarks extracted successfully!" << std::endl;

        // 6. Draw and Save
        cv::Mat result_img = draw_landmarks(img, landmarks, SCORE_THRESHOLD);
        std::string out_path = "blazepose_landmark_result/" + it.path().filename().string();
        cv::imwrite(out_path, result_img);
        std::cout << "Result saved to: " << out_path << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}
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
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

const float PRESENCE_THRESHOLD = 0.5f;
const float VISIBILITY_THRESHOLD = 0.5f;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir> <detections_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::string detections_dir = argv[3];

    std::cout << "BlazePose Landmark Demo" << std::endl;
    fs::create_directory("blazepose_landmark_result");

    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    if (io_num.n_output != 5)
    {
        std::cerr << "Expected 5 BlazePose landmark outputs, got " << io_num.n_output << std::endl;
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

    std::cout << "Input shape: [" << input_height << ", " << input_width << ", 3]" << std::endl;

    if (input_height != 256 || input_width != 256)
    {
        std::cerr << "Expected a 256x256 BlazePose landmark input." << std::endl;
        uninit_network(context);
        return -1;
    }

    std::vector<std::vector<int>> out_shapes;

    for (int i = 0; i < io_num.n_output; ++i)
        out_shapes.push_back(get_tensor_shape(query_output_attr(context, i)));

    std::vector<amlnn_output> outData(io_num.n_output);

    for (auto &it : fs::directory_iterator(argv[2]))
    {
        if (!it.is_regular_file())
            continue;

        cv::Mat image = cv::imread(it.path().string());

        if (image.empty())
            continue;

        std::string filename = it.path().stem().string();
        std::string txt_path = (fs::path(detections_dir) / (filename + "_det.txt")).string();
        std::vector<Detection> detections = load_detections(txt_path);

        if (detections.empty())
        {
            std::cout << "No detections found for " << filename << ", skipping..." << std::endl;
            continue;
        }

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        std::vector<PoseResult> results;
        double total_inference_time = 0.0;

        for (const auto &detection : detections)
        {
            Roi roi = detection_to_roi(detection, image.cols, image.rows);
            cv::Mat preprocessed = preprocess(image, roi, {input_height, input_width});
            std::vector<uint8_t> prepared_data = prepare_input_tensor(preprocessed, input_attr);

            if (prepared_data.empty())
            {
                std::cerr << "Failed to prepare input tensor." << std::endl;
                continue;
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            if (!run_network(context, prepared_data.data(), prepared_data.size(), outData))
            {
                std::cerr << "Failed to run network." << std::endl;
                uninit_network(context);
                return -1;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
            total_inference_time += inference_time.count();

            std::vector<float *> out_ptrs;

            for (int i = 0; i < io_num.n_output; ++i)
                out_ptrs.push_back(static_cast<float *>(outData[i].buf));

            PoseResult result;

            if (postprocess(out_ptrs, out_shapes, roi, image.cols, image.rows, PRESENCE_THRESHOLD, result))
                results.push_back(result);
        }

        std::cout << "Poses: " << results.size() << std::endl;
        std::cout << "Total landmark inference time: " << total_inference_time << " ms" << std::endl;

        cv::Mat result_image = draw_detections(image, results, VISIBILITY_THRESHOLD);
        std::string result_path = "blazepose_landmark_result/" + filename + "_result.jpg";
        std::string landmark_path = "blazepose_landmark_result/" + filename + "_landmarks.txt";

        cv::imwrite(result_path, result_image);
        save_landmarks(landmark_path, results);

        std::cout << "Result saved to: " << result_path << std::endl;
        std::cout << "Landmarks saved to: " << landmark_path << std::endl;
    }

    std::cout << "============================================================" << std::endl << std::endl;
    uninit_network(context);
    return 0;
}
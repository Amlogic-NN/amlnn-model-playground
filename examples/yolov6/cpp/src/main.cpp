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

const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.2f;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir> [labels.txt (optional)]\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::string labels_path = "../input/coco_80_names.txt";

    if (argc >= 4)
    {
        labels_path = argv[3];
    }

    std::cout << "YOLOv6 Demo" << std::endl;

    if (!fs::exists(labels_path))
    {
        std::cerr << "Warning: Labels file not found at path: " << labels_path << ". Using fallback IDs." << std::endl;
    }

    std::string output_dir = "yolov6_result";
    fs::create_directory(output_dir);

    std::vector<std::string> labels = load_labels(labels_path);

    // 1. Initialize Network
    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    // Query IO numbers to ensure we have exactly 6 outputs
    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_output != 6)
    {
        std::cerr << "Warning: Expected 6 outputs (3 cls, 3 bbox heads), but model has "
                  << io_num.n_output << " outputs." << std::endl;
    }

    // Query Input Attribute for Scale and Zero Point
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
        auto [preprocessed, scale, pad] = preprocess(img, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));
        cv::Mat quantized_img = quantize_input(preprocessed, input_attr.scale, input_attr.zp);

        // 4. Set input, run inference, and Get Outputs
        size_t input_size = input_attr.n_elems * sizeof(int8_t);

        auto start_time = std::chrono::high_resolution_clock::now();
        if (!run_network(context, quantized_img.data, input_size, outData))
        {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }

        if (outData.empty())
            return -1;

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        // Prepare pointers for the 6 outputs
        std::vector<float*> out_ptrs;
        for (int i = 0; i < io_num.n_output; ++i) {
            out_ptrs.push_back((float*)outData[i].buf);
        }

        // 5. Postprocess
        std::vector<Detection> detections = postprocess(
            out_ptrs, out_shapes,
            std::make_tuple(preprocessed, scale, pad),
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            MODEL_INPUT_WIDTH);

        std::cout << "Detections after NMS: " << detections.size() << std::endl;

        // 6. Draw and Save
        cv::Mat result_img = draw_detections(img, detections, labels);
        std::string out_path = output_dir + "/" + it.path().filename().string();
        cv::imwrite(out_path, result_img);
        std::cout << "Result saved to: " << out_path << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}
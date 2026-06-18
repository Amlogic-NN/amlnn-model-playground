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
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 480;
const float SCORE_THRESHOLD = 0.3f;
const float NMS_THRESHOLD = 0.45f;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];

    std::cout << "YOLO-World Demo" << std::endl;
    fs::create_directory("yoloworld_result");

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
    int n_outputs = io_num.n_output;

    // Query Input Attributes dynamically
    amlnn_tensor_attr input_attr = query_input_attr(context, 0);
    size_t input_size = input_attr.n_elems * sizeof(uint8_t);

    // Prepare outputs vector
    std::vector<amlnn_output> outData(n_outputs);

    // 2. Loop through all images in directory
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
        auto [preprocessed, scale, pad] = preprocess(img, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));
        cv::Mat quantized_img = quantize_input(preprocessed, input_attr.scale, input_attr.zp, input_attr.type);

        // 4. Run Inference
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(context, quantized_img.data, input_size, outData))
        {
            std::cerr << "Failed to run network" << std::endl;
            continue;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        // 5. Gather and Sort Outputs
        std::vector<OutputLayer> layers;
        for (int i = 0; i < n_outputs; i++)
        {
            // Assign to local variable first
            amlnn_tensor_attr attr = query_output_attr(context, i);
            std::vector<int> shape = get_tensor_shape(attr);

            // STRICT 77-Channel Area Calculation
            int total_elems = 1;
            for (int d : shape)
                total_elems *= d;
            int area = total_elems / 77;

            layers.push_back({(float *)outData[i].buf, shape, area});
        }

        // Sort descending by area (Largest grid = smallest stride)
        std::sort(layers.begin(), layers.end(), [](const OutputLayer &a, const OutputLayer &b)
                  { return a.area > b.area; });

        std::vector<float *> out_buffers;
        std::vector<std::vector<int>> out_shapes;
        for (const auto &layer : layers)
        {
            out_buffers.push_back(layer.buf);
            out_shapes.push_back(layer.shape);
        }

        // 6. Postprocess
        std::vector<Detection> detections = postprocess(
            out_buffers, out_shapes,
            std::make_tuple(preprocessed, scale, pad),
            SCORE_THRESHOLD, NMS_THRESHOLD);

        std::cout << "Detections: " << detections.size() << std::endl;
        for (size_t j = 0; j < detections.size(); j++)
        {
            std::cout << "  " << j + 1 << ". Class " << WORLD_CLASSES[detections[j].class_id]
                      << " - Score: " << detections[j].score << "\n";
        }
        std::cout << std::endl;

        // 7. Draw and Save
        cv::Mat result_img = draw_detections(img, detections);
        std::string out_path = "yoloworld_result/" + it.path().filename().string();
        cv::imwrite(out_path, result_img);

        std::cout << "Result saved to: " << out_path << std::endl
                  << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 8. Cleanup
    uninit_network(context);
    return 0;
}
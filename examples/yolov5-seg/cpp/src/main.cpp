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
const int MODEL_INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5f;
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

    std::cout << "YOLOv5-Seg Object Detection Demo" << std::endl;

    fs::create_directory("yolov5_seg_result");

    // 1. Initialize Network
    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    // Query IO numbers to ensure we have 4 outputs for YOLO-seg (3 bounding box strides + 1 mask proto)
    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_output != 4)
    {
        std::cerr << "Warning: Expected 4 outputs, but model has "
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

    // Allocate Outputs
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

        float *branch_data[3] = {
            (float *)outData[0].buf,
            (float *)outData[1].buf,
            (float *)outData[2].buf};

        std::vector<int> branch_shapes[3] = {
            out_shapes[0],
            out_shapes[1],
            out_shapes[2]};

        float *proto_data = (float *)outData[3].buf;
        std::vector<int> proto_shape = out_shapes[3];

        // 6. Postprocess
        std::vector<Detection> detections = postprocess(
            branch_data[0], branch_shapes[0],
            branch_data[1], branch_shapes[1],
            branch_data[2], branch_shapes[2],
            std::make_tuple(preprocessed, scale, pad),
            SCORE_THRESHOLD,
            NMS_THRESHOLD);

        std::cout << "Detections after NMS: " << detections.size() << std::endl;

        // 7. Draw Segmentations and Bounding Boxes
        cv::Mat result_img = draw_detections(img, detections, proto_data, proto_shape, scale, pad);

        std::string out_path = "yolov5_seg_result/" + it.path().filename().string();
        cv::imwrite(out_path, result_img);
        std::cout << "Result saved to: " << out_path << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // 8. Cleanup
    uninit_network(context);

    return 0;
}
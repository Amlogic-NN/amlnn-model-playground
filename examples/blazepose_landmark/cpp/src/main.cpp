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
#include "postprocess.h"
#include "model_loader.h"

const std::string DEFAULT_OUTPUT_PATH = "./result.jpg";
const int MODEL_INPUT_WIDTH = 256;
const int MODEL_INPUT_HEIGHT = 256;
const float SCORE_THRESHOLD = 0.5f;

int main(int argc, char **argv)
{
    std::string model_path;
    std::string image_path;

    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    if (argc > 1)
        model_path = argv[1];
    if (argc > 2)
        image_path = argv[2];

    std::cout << "Blazepose Detect Demo" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Output: " << DEFAULT_OUTPUT_PATH << std::endl;

    // 1. Load Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        std::cerr << "Failed to load image from " << image_path << std::endl;
        return -1;
    }
    // Load detections
    // n * 13 detections
    // image_path -> txt_path
    std::vector<std::vector<float>> detections;

    std::string txt_path = image_path.substr(0, image_path.find_last_of('.'));
    txt_path += ".txt";
    std::ifstream ifs(txt_path);
    for (std::string line; std::getline(ifs, line);)
    {
        std::istringstream iss(line);
        std::vector<float> det;
        float val;
        while (iss >> val)
            det.push_back(val);
        if (!det.empty())
            detections.push_back(det);
    }

    // 2. Initialize Network
    void *context = init_network(model_path.c_str());
    if (!context)
    {
        std::cerr << "Failed to initialize network." << std::endl;
        return -1;
    }

    // 3. Preprocess
    auto start_time = std::chrono::high_resolution_clock::now();

    auto [preprocessed, affine] = preprocess(img, detections, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));

    // Quantize to int16 (model expects quantized input)
    cv::Mat quantized_img = quantize_input(preprocessed, 0.000030518509447574615f);

    // 4. Set input and run inference
    nn_input inData;
    memset(&inData, 0, sizeof(nn_input));
    inData.input_type = BINARY_RAW_DATA;
    inData.input = quantized_img.data;
    inData.input_index = 0;
    inData.size = quantized_img.total() * quantized_img.elemSize();

    if (aml_module_input_set(context, &inData) != 0)
    {
        std::cerr << "Failed to set input." << std::endl;
        uninit_network(context);
        return -1;
    }

    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.format = AML_OUTDATA_FLOAT32;

    nn_output *outdata = (nn_output *)aml_module_output_get(context, outconfig);
    if (!outdata)
    {
        std::cerr << "Failed to run network." << std::endl;
        uninit_network(context);
        return -1;
    }

    // 5. Postprocess
    std::vector<BlazePoseLandmark> landmarks = postprocess(outdata, affine);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;

    std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
    std::cout << "Landmarks: " << landmarks.size() << std::endl;

    // 6. Draw and Save
    cv::Mat result_img = draw_landmarks(img, landmarks, SCORE_THRESHOLD);
    cv::imwrite(DEFAULT_OUTPUT_PATH, result_img);
    std::cout << "Result saved to " << DEFAULT_OUTPUT_PATH << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}

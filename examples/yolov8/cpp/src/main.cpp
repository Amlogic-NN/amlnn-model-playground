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
#include <opencv2/opencv.hpp>
#include "postprocess.h"
#include "model_loader.h"

const std::string DEFAULT_OUTPUT_PATH = "./result.jpg";
const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;

int main(int argc, char** argv) {
    std::string model_path;
    std::string image_path;

    if (argc != 3) {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    if (argc > 1) model_path = argv[1];
    if (argc > 2) image_path = argv[2];

    std::cout << "YOLOv8 Demo" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Output: " << DEFAULT_OUTPUT_PATH << std::endl;

    // 1. Load Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image from " << image_path << std::endl;
        return -1;
    }

    // 2. Initialize Network
    void* context = init_network(model_path.c_str());
    if (!context) {
        std::cerr << "Failed to initialize network." << std::endl;
        return -1;
    }

    // 3. Preprocess
    auto start_time = std::chrono::high_resolution_clock::now();

    auto [preprocessed, scale, pad] = preprocess(img, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));

    // Quantize to int8 (model expects quantized input)
    cv::Mat quantized_img = quantize_input(preprocessed);

    // 4. Set input and run inference
    nn_input inData;
    memset(&inData, 0, sizeof(nn_input));
    inData.input_type = BINARY_RAW_DATA;
    inData.input = quantized_img.data;
    inData.input_index = 0;
    inData.size = quantized_img.total() * quantized_img.elemSize();

    if (aml_module_input_set(context, &inData) != 0) {
        std::cerr << "Failed to set input." << std::endl;
        uninit_network(context);
        return -1;
    }

    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.format = AML_OUTDATA_FLOAT32;

    nn_output* outdata = (nn_output*)aml_module_output_get(context, outconfig);
    if (!outdata) {
        std::cerr << "Failed to run network." << std::endl;
        uninit_network(context);
        return -1;
    }

    // 5. Postprocess
    float* outbuf0 = (float*)outdata->out[0].buf;
    float* outbuf1 = (float*)outdata->out[1].buf;
    float* outbuf2 = (float*)outdata->out[2].buf;

    const int channels = 144;  // 64 DFL + 80 classes
    
    std::vector<Detection> detections = postprocess(
        std::make_tuple(outbuf0, std::make_tuple(MODEL_INPUT_HEIGHT / 16, MODEL_INPUT_WIDTH / 16, channels), 16),
        std::make_tuple(outbuf1, std::make_tuple(MODEL_INPUT_HEIGHT / 8, MODEL_INPUT_WIDTH / 8, channels), 8),
        std::make_tuple(outbuf2, std::make_tuple(MODEL_INPUT_HEIGHT / 32, MODEL_INPUT_WIDTH / 32, channels), 32),
        std::make_tuple(preprocessed, scale, pad),
        SCORE_THRESHOLD,
        NMS_THRESHOLD
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;

    std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
    std::cout << "Detections: " << detections.size() << std::endl;

    // 6. Draw and Save
    cv::Mat result_img = draw_detections(img, detections);
    cv::imwrite(DEFAULT_OUTPUT_PATH, result_img);
    std::cout << "Result saved to " << DEFAULT_OUTPUT_PATH << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}

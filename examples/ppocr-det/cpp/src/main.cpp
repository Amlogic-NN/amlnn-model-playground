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
#include <opencv2/opencv.hpp>
#include <chrono>
#include "nn_sdk.h"
#include "model_loader.h"
#include "postprocess.h"

const std::string DEFAULT_OUTPUT_PATH = "result.png";

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model_path> <image_path> [output_path]\n", argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string output_path = (argc > 3) ? argv[3] : DEFAULT_OUTPUT_PATH;

    printf("Model: %s\n", model_path.c_str());
    printf("Image: %s\n", image_path.c_str());

    // 1. Initialize Network
    void* ctx = init_network(model_path.c_str());
    if (!ctx) {
        fprintf(stderr, "Failed to initialize network\n");
        return -1;
    }

    // 2. Load Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        fprintf(stderr, "Failed to load image: %s\n", image_path.c_str());
        uninit_network(ctx);
        return -1;
    }

    // 3. Preprocess
    auto start_time = std::chrono::high_resolution_clock::now();
    cv::Mat pre_image;
    float scale = 1.0f;
    preprocess(img, pre_image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, scale);

    printf("scale: %f\n", scale);
    // 4. Inference
    nn_output* outdata = (nn_output*)run_paddleocr_network(ctx, pre_image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_CHANNELS);
    if (!outdata) {
        fprintf(stderr, "Inference failed\n");
        uninit_network(ctx);
        return -1;
    }

    // 5. Postprocess
    float* out0 = (float*)outdata->out[0].buf;

    std::vector<Object> results;
    postprocess(out0, img, BOX_SCORE_THRESH, BOX_THRESH, results, scale);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
    printf("Inference + Postprocess time: %.2f ms\n", inference_time.count());
    printf("Results: %zu\n", results.size());

    // 6. Draw and Save
    cv::Mat res = draw_objects(img, results);
    cv::imwrite(output_path, res);
    printf("Saved result to %s\n", output_path.c_str());

    // 7. Cleanup
    uninit_network(ctx);

    return 0;
}

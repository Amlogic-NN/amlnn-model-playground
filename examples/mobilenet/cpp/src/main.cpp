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
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include "nn_sdk.h"
#include "model_loader.h"

const int MODEL_INPUT_WIDTH = 224;
const int MODEL_INPUT_HEIGHT = 224;
const int TOP_K = 5;

typedef struct {
    int index;
    float score;
} ClassScore;

bool compare_scores(const ClassScore& a, const ClassScore& b) {
    return a.score > b.score;
}

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(line.find_last_not_of(" \n\r\t")+1);
            labels.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Warning: Could not open labels file: " << path << std::endl;
    }
    return labels;
}



int main(int argc, char** argv) {
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string labels_path = argv[3];

    std::cout << "MobileNetV2 Demo" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Labels: " << labels_path << std::endl;

    // 1. Initialize Network
    void* context = init_network(model_path.c_str());
    if (!context) {
        std::cerr << "Failed to initialize network." << std::endl;
        return -1;
    }

    // 2. Load and Preprocess Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image from " << image_path << std::endl;
        uninit_network(context);
        return -1;
    }

    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT));

    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);

    nn_input inData;
    memset(&inData, 0, sizeof(nn_input));
    inData.input_type = BINARY_RAW_DATA;
    inData.input = img_rgb.data;
    inData.input_index = 0;
    inData.size = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3 * sizeof(unsigned char);

    if (aml_module_input_set(context, &inData) != 0) {
         std::cerr << "Failed to set input." << std::endl;
         uninit_network(context);
         return -1;
    }

    // 3. Run Inference
    auto start_time = std::chrono::high_resolution_clock::now();

    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.format = AML_OUTDATA_FLOAT32;
    
    nn_output* outdata = (nn_output*)aml_module_output_get(context, outconfig);
    if (!outdata) {
         std::cerr << "Failed to run network (get output)." << std::endl;
         uninit_network(context);
         return -1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
    std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

    // 4. Postprocess
    if (outdata->num <= 0) {
        std::cerr << "No output from network." << std::endl;
        uninit_network(context);
        return -1;
    }

    float* output_buffer = (float*)outdata->out[0].buf;
    int num_classes = outdata->out[0].size / sizeof(float);

    std::vector<ClassScore> scores;
    for (int i = 0; i < num_classes; ++i) {
        scores.push_back({i, output_buffer[i]});
    }

    std::sort(scores.begin(), scores.end(), compare_scores);

    std::vector<std::string> labels = load_labels(labels_path);

    std::cout << "\nTop-" << TOP_K << " Classification Results:" << std::endl;
    for (int i = 0; i < std::min(TOP_K, num_classes); ++i) {
        int idx = scores[i].index;
        float score = scores[i].score;
        std::string label = (idx < labels.size()) ? labels[idx] : "Class " + std::to_string(idx);
        printf("  %d. %-20s (score: %.6f)\n", i + 1, label.c_str(), score);
    }

    // 5. Cleanup
    uninit_network(context);

    return 0;
}

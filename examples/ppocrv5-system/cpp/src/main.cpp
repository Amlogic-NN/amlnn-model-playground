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
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "postprocess.h"

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0]
                  << " --image_path=<image_path>"
                  << " --det_model_path=<det_model_path>"
                  << " --rec_model_path=<rec_model_path>"
                  << " --dict_path=<dict_path>\n";
        return -1;
    }

    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto pos = arg.find('=');
        if (pos != std::string::npos) {
            args[arg.substr(0, pos)] = arg.substr(pos + 1);
        }
    }

    if (!args.count("--image_path") ||
        !args.count("--det_model_path") ||
        !args.count("--rec_model_path") ||
        !args.count("--dict_path")) {

        std::cerr << "Error: missing required arguments\n";
        return -1;
    }

    std::string image_path     = args["--image_path"];
    std::string det_model_path = args["--det_model_path"];
    std::string rec_model_path = args["--rec_model_path"];
    std::string dict_path      = args["--dict_path"];

    std::cout << "image: " << image_path << "\n"
              << "det model: " << det_model_path << "\n"
              << "rec model: " << rec_model_path << "\n"
              << "dict: " << dict_path << "\n";

    try {
        OcrEngine ocr_engine(det_model_path, rec_model_path, dict_path);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: failed to load image: " << image_path << std::endl;
            return -1;
        }

        auto results = ocr_engine.Process(image);

        OcrUtils utils;
        cv::Mat result_image = image.clone();
        utils.DrawOCRResults(result_image, results, 0.3f);

        std::string output_path = "ocr_result_" + std::filesystem::path(image_path).filename().string();
        if (!cv::imwrite(output_path, result_image)) {
            std::cerr << "Error: failed to save result image to " << output_path << "\n";
            return -1;
        }
        std::cout << "OCR results saved to: " << output_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during OCR processing: " << e.what() << std::endl;
    }

    return 0;
}
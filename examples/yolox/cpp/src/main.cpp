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
#include <cmath>
#include <opencv2/opencv.hpp>
#include "postprocess.h"
#include "model_loader.h"

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

const std::string DEFAULT_OUTPUT_PATH = "./result.jpg";
const int MODEL_INPUT_WIDTH = 640;
const int MODEL_INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
const float CONF_THRESHOLD = 0.45f;

const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model_path> <image_path> [output_path]\n", argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string output_path = (argc > 3) ? argv[3] : DEFAULT_OUTPUT_PATH;

    std::cout << "YOLOX C++ Demo" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;

    // 1. Load Image
    cv::Mat origin_img = cv::imread(image_path);
    if (origin_img.empty()) {
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
    cv::Mat img;
    float scale;
    std::tuple<int, int> pad;
    std::tie(img, scale, pad) = preproc(origin_img, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));
    int pad_left = std::get<0>(pad);
    int pad_top = std::get<1>(pad);

    // 4. Run Network
    std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple = 
        std::make_tuple(img, scale, pad);

    auto start_time = std::chrono::high_resolution_clock::now();

    void* output_ptr = run_network(context, {input_tuple});
    if (!output_ptr) {
        std::cerr << "Failed to run network." << std::endl;
        uninit_network(context);
        return -1;
    }
    nn_output* outdata = (nn_output*)output_ptr;

    // 5. Postprocess
    int num_classes = CLASS_NAMES.size();
    std::vector<Detection> detections;

    if (outdata->num == 1) {
        // Single output YOLOX model [1, 8400, 85]
        float* output = (float*)outdata->out[0].buf;

        int num_boxes = 8400;  // Default for YOLOX
        if (outdata->out[0].param && outdata->out[0].param->num_of_dims >= 2) {
            if (outdata->out[0].param->num_of_dims == 3) {
                num_boxes = outdata->out[0].param->sizes[1];
            } else if (outdata->out[0].param->num_of_dims == 2) {
                num_boxes = outdata->out[0].param->sizes[0];
            }
        }

        demo_postprocess(output, num_boxes, std::make_tuple(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), false);

        // boxes/scores
        std::vector<cv::Rect2f> boxes;
        std::vector<std::vector<float>> scores;
        extract_boxes_and_scores(
            output, num_boxes, num_classes,
            scale, pad_left, pad_top,
            origin_img.cols, origin_img.rows,
            boxes, scores
        );

        // multiclass_nms
        detections = multiclass_nms(boxes, scores, num_classes, NMS_THRESHOLD, 0.1f);
    } else {
        std::cerr << "Error: Unsupported output count: " << outdata->num << std::endl;
        uninit_network(context);
        return -1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
    std::cout << "Inference + Postprocess time: " << inference_time.count() << " ms" << std::endl;
    std::cout << "Detections found: " << detections.size() << std::endl;

    // 6. Visualize and Save
    cv::Mat result_img = vis(origin_img, detections, CONF_THRESHOLD, CLASS_NAMES);
    cv::imwrite(output_path, result_img);
    std::cout << "Result saved to " << output_path << std::endl;

    // 7. Cleanup
    uninit_network(context);

    return 0;
}


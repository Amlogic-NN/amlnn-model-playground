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

#ifndef _AMLNN_YOLOV8_DEMO_POSTPROCESS_H_
#define _AMLNN_YOLOV8_DEMO_POSTPROCESS_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>

// Detection result structure
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float score;           // Confidence score
    int class_id;          // Predicted class ID
};

// COCO class names (80 classes)
extern const char* COCO_CLASSES[80];

// Preprocess image with letterbox resizing
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);

// Quantize float32 image to int8 for model input
cv::Mat quantize_input(const cv::Mat& float_img, float scale = 0.003921568859368563f, int8_t zero_point = -128);

// Postprocess YOLOv8 outputs with DFL decoding
std::vector<Detection> postprocess(std::tuple<float*, std::tuple<int, int, int>, int> out0,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out1,
                                   std::tuple<float*, std::tuple<int, int, int>, int> out2,
                                   std::tuple<cv::Mat, float, std::tuple<int, int>> input_tuple,
                                   float conf_thresh, float iou_threshold);

// Draw detections on image
cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& detections);

#endif // _AMLNN_YOLOV8_DEMO_POSTPROCESS_H_

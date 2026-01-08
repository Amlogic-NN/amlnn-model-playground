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

#ifndef YOLOX_POSTPROCESS_H
#define YOLOX_POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>

/**
 * YOLOX preprocessing function
 * @param img Input image (BGR format)
 * @param input_size Target size (height, width)
 * @return Processed image (HWC format, float32, ImageNet normalized, RGB format), scale factor, padding (left, top)
 * Note: NNSDK's model_loader expects HWC format, so return HWC instead of CHW
 * Processing steps:
 * 1. letterbox (resize + padding with 114)
 * 2. BGR to RGB conversion
 * 3. Normalize to 0-1 (divide by 255.0)
 * 4. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 */
std::tuple<cv::Mat, float, std::tuple<int, int>> preproc(const cv::Mat& img, std::tuple<int, int> input_size);

// Detection result structure
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float score;           // Confidence score
    int class_id;          // Predicted class ID
};

/**
 * YOLOX official demo_postprocess function (C++ implementation)
 * Decode model output to absolute coordinates
 * @param outputs Model output [batch, num_boxes, 85]
 * @param num_boxes Number of detection boxes
 * @param img_size Input image size (height, width)
 * @param p6 Whether to use P6 (default false, use P5)
 * @return Decoded output [num_boxes, 85], format: [cx, cy, w, h, obj_conf, class0, ..., class79]
 */
void demo_postprocess(float* outputs, int num_boxes, std::tuple<int, int> img_size, bool p6 = false);

/**
 * Extract boxes and scores from output after demo_postprocess
 * @param output Model output (processed by demo_postprocess) [num_boxes * 85]
 * @param num_boxes Number of detection boxes
 * @param num_classes Number of classes
 * @param scale Scale factor from preprocessing
 * @param pad_left Left padding boundary
 * @param pad_top Top padding boundary
 * @param img_width Original image width
 * @param img_height Original image height
 * @param boxes Output boxes (xyxy format, mapped to original image size)
 * @param scores Output scores (class scores for each box, obj_conf * cls_scores)
 */
void extract_boxes_and_scores(
    float* output,
    int num_boxes,
    int num_classes,
    float scale,
    int pad_left,
    int pad_top,
    int img_width,
    int img_height,
    std::vector<cv::Rect2f>& boxes,
    std::vector<std::vector<float>>& scores
);

/**
 * Single-class NMS
 */
std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float nms_thr);

/**
 * YOLOX official multiclass_nms function (class-agnostic version)
 * @param boxes Detection boxes [N, 4] (x1, y1, x2, y2)
 * @param scores Class scores [N, num_classes]
 * @param num_classes Number of classes
 * @param nms_thr NMS threshold
 * @param score_thr Score threshold
 * @return Detection results, each row is [x1, y1, x2, y2, score, class_id]
 */
std::vector<Detection> multiclass_nms(const std::vector<cv::Rect2f>& boxes, 
                                      const std::vector<std::vector<float>>& scores,
                                      int num_classes,
                                      float nms_thr, 
                                      float score_thr);

/**
 * Visualize detection results (consistent with Python version, supports adaptive font size)
 */
cv::Mat vis(const cv::Mat& img, 
            const std::vector<Detection>& detections,
            float conf_thresh,
            const std::vector<std::string>& class_names);

#endif // YOLOX_POSTPROCESS_H


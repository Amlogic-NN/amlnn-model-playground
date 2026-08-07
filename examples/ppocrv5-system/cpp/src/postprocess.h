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

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

struct DetectionResult {
    std::vector<cv::Point2f> points;
    float score;
};

struct OCRResult {
    DetectionResult box;
    std::string text;
    float score;
};

class TextDetector {
public:
    TextDetector(const fs::path model_path);
    int InitNetwork();
    int UninitNetwork();
    std::vector<DetectionResult> Detect(const cv::Mat &image);
private:
    void* ctx_ = nullptr;
    fs::path model_path_;

    float resize_scale_x_;
    float resize_scale_y_;
    int resized_w_;
    int resized_h_;

    cv::Mat Preprocess(const cv::Mat &image);
    std::vector<DetectionResult> Postprocess(const cv::Mat& image, const std::vector<float>& data);
    std::pair<std::vector<cv::Point2f>, float> GetMiniBoxes(const std::vector<cv::Point>& contour);
    float BoxScoreFast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box);
    std::vector<cv::Point2f> UnclipPolygon(const std::vector<cv::Point2f>& box, float unclip_ratio);
};

class TextRecognizer {
public:
    TextRecognizer(const fs::path model_path, const fs::path dict_path);
    int InitNetwork();
    int UninitNetwork();
    std::string Recognize(const cv::Mat &image);
private:
    void* ctx_ = nullptr;
    fs::path model_path_;
    fs::path dict_path_;
    std::vector<std::string> character_dict_;

    bool LoadCharacterDict(const std::string& dict_path);
    cv::Mat Preprocess(const cv::Mat &image);
    std::pair<std::string, float> Postprocess(const std::vector<float>& output_data);
};

class OcrEngine {
public:
    OcrEngine(const fs::path& det_model_path, const fs::path& rec_model_path, const fs::path& dict_path);
    std::vector<OCRResult> Process(const cv::Mat& image);
private:
    TextDetector detector_;
    TextRecognizer recognizer_;

    std::vector<DetectionResult> SortBoxes(const std::vector<DetectionResult>& boxes);
    cv::Mat GetRotateCropImage(const cv::Mat& image, const DetectionResult& box);
};


class OcrUtils {
public:
    int DrawBoxes(cv::Mat& image, const std::vector<DetectionResult> &boxes, float drop_score = 0.5f);
    int DrawOCRResults(cv::Mat& image, const std::vector<OCRResult> &result, float drop_score = 0.5f);
};
#ifndef YOLO11_POSTPROCESS_H
#define YOLO11_POSTPROCESS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

static constexpr int kInputW = 640;
static constexpr int kInputH = 640;
static constexpr int kNumClasses = 80;
static constexpr int kDflChannels = 16;
static constexpr int kTotalChannels = 144;

extern const std::vector<std::string> kClassNames;

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

float decode_dfl(const float* dfl_ptr);
float calculate_iou(const cv::Rect& a, const cv::Rect& b);
std::vector<int> manual_nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float thresh);

#endif
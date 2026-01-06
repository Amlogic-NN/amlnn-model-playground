#ifndef RESNET_POSTPROCESS_H
#define RESNET_POSTPROCESS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

static const float MEAN[3] = {123.675f, 116.28f, 103.53f};
static const float STD[3]  = {58.395f, 58.395f, 58.395f};
static constexpr int kInputW = 224;
static constexpr int kInputH = 224;

void preprocess(const cv::Mat& src, float* dst);

void postprocess_topk(float* logits, int size, const std::vector<std::string>& labels, int k = 5);

std::vector<std::string> load_labels(const std::string& path);

#endif
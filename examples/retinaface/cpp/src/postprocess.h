#ifndef RETINAFACE_POSTPROCESS_H
#define RETINAFACE_POSTPROCESS_H

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

static constexpr int kInputW = 320;
static constexpr int kInputH = 320;

std::vector<std::array<float, 4>> generate_priors();

std::array<float, 10> decode_landm(const float* lm, int idx, int total, bool is_planar, const std::array<float, 4>& p);

std::array<float, 4> decode_box(const float* loc, int idx, int total, bool is_planar, const std::array<float, 4>& p);

float iou(const std::array<float, 4>& a, const std::array<float, 4>& b);

std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes, const std::vector<float>& scores, float thresh);

#endif
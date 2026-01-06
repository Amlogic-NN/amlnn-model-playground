#include "postprocess.h"
#include <cmath>
#include <numeric>
#include <algorithm>

std::vector<std::array<float, 4>> generate_priors() {
    std::vector<std::array<float, 4>> priors;
    std::vector<int> steps = {8, 16, 32};
    std::vector<std::vector<int>> min_sizes = {{16, 32}, {64, 128}, {256, 512}};
    for (size_t k = 0; k < steps.size(); ++k) {
        int fm_h = std::ceil((float)kInputH / steps[k]);
        int fm_w = std::ceil((float)kInputW / steps[k]);
        for (int i = 0; i < fm_h; i++) {
            for (int j = 0; j < fm_w; j++) {
                for (int ms : min_sizes[k]) {
                    float cx = (j + 0.5f) * steps[k] / kInputW;
                    float cy = (i + 0.5f) * steps[k] / kInputH;
                    float sx = (float)ms / kInputW;
                    float sy = (float)ms / kInputH;
                    priors.push_back({cx, cy, sx, sy});
                }
            }
        }
    }
    return priors;
}

std::array<float, 10> decode_landm(const float* lm, int idx, int total, bool is_planar, const std::array<float, 4>& p) {
    std::array<float, 10> out{};
    float raw[10];
    if (is_planar) {
        for (int j = 0; j < 10; ++j) raw[j] = lm[j * total + idx];
    } else {
        for (int j = 0; j < 10; ++j) raw[j] = lm[idx * 10 + j];
    }
    for (int i = 0; i < 5; i++) {
        out[2 * i]     = p[0] + raw[2 * i]     * 0.1f * p[2];
        out[2 * i + 1] = p[1] + raw[2 * i + 1] * 0.1f * p[3];
    }
    return out;
}

std::array<float, 4> decode_box(const float* loc, int idx, int total, bool is_planar, const std::array<float, 4>& p) {
    float l[4];
    if (is_planar) {
        l[0] = loc[0 * total + idx]; l[1] = loc[1 * total + idx];
        l[2] = loc[2 * total + idx]; l[3] = loc[3 * total + idx];
    } else {
        l[0] = loc[idx * 4 + 0]; l[1] = loc[idx * 4 + 1];
        l[2] = loc[idx * 4 + 2]; l[3] = loc[idx * 4 + 3];
    }
    float cx = p[0] + l[0] * 0.1f * p[2];
    float cy = p[1] + l[1] * 0.1f * p[3];
    float w = p[2] * std::exp(l[2] * 0.2f);
    float h = p[3] * std::exp(l[3] * 0.2f);
    return { cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f };
}

float iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float xx1 = std::max(a[0], b[0]), yy1 = std::max(a[1], b[1]);
    float xx2 = std::min(a[2], b[2]), yy2 = std::min(a[3], b[3]);
    float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float areaA = (a[2] - a[0]) * (a[3] - a[1]), areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (areaA + areaB - inter);
}

std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes, const std::vector<float>& scores, float thresh) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0]; keep.push_back(i); order.erase(order.begin());
        order.erase(std::remove_if(order.begin(), order.end(), [&](int j) { return iou(boxes[i], boxes[j]) > thresh; }), order.end());
    }
    return keep;
}
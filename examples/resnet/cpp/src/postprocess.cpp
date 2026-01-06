#include "postprocess.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>

void preprocess(const cv::Mat& src, float* dst) {
    cv::Mat rgb, resized;
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(kInputW, kInputH));

    for (int i = 0; i < kInputH; ++i) {
        for (int j = 0; j < kInputW; ++j) {
            cv::Vec3f pixel = resized.at<cv::Vec3b>(i, j);
            int idx = (i * kInputW + j) * 3;
            dst[idx + 0] = (pixel[0] - MEAN[0]) / STD[0];
            dst[idx + 1] = (pixel[1] - MEAN[1]) / STD[1];
            dst[idx + 2] = (pixel[2] - MEAN[2]) / STD[2];
        }
    }
}

void postprocess_topk(float* logits, int size, const std::vector<std::string>& labels, int k) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    std::cout << "\nTop-" << k << " Results:" << std::endl;
    for (int i = 0; i < k; ++i) {
        int idx = indices[i];
        std::string name = (idx < (int)labels.size()) ? labels[idx] : "N/A";
        printf("%d: %-20s  score=%.6f\n", i + 1, name.c_str(), logits[idx]);
    }
}

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if(!line.empty()) labels.push_back(line);
    }
    return labels;
}
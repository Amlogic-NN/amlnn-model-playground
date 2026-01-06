#include "postprocess.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <float.h>

const std::vector<std::string> kClassNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

float decode_dfl(const float* dfl_ptr) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < kDflChannels; i++) {
        if (dfl_ptr[i] > max_val) max_val = dfl_ptr[i];
    }
    float exp_sum = 0;
    float exp_vals[kDflChannels];
    for (int i = 0; i < kDflChannels; i++) {
        exp_vals[i] = std::exp(dfl_ptr[i] - max_val);
        exp_sum += exp_vals[i];
    }
    float res = 0;
    for (int i = 0; i < kDflChannels; i++) {
        res += (exp_vals[i] / exp_sum) * i;
    }
    return res;
}

float calculate_iou(const cv::Rect& a, const cv::Rect& b) {
    int xx1 = std::max(a.x, b.x);
    int yy1 = std::max(a.y, b.y);
    int xx2 = std::min(a.x + a.width, b.x + b.width);
    int yy2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);
    float inter = (float)w * h;
    float areaA = (float)a.width * a.height;
    float areaB = (float)b.width * b.height;
    return inter / (areaA + areaB - inter);
}

std::vector<int> manual_nms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float thresh) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { 
        return scores[a] > scores[b]; 
    });
    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);
        order.erase(order.begin());
        order.erase(std::remove_if(order.begin(), order.end(), [&](int j) { 
            return calculate_iou(boxes[i], boxes[j]) > thresh; 
        }), order.end());
    }
    return keep;
}
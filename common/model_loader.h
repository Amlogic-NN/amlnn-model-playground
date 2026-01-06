#ifndef _AMLNN_MODEL_LOADER_H_
#define _AMLNN_MODEL_LOADER_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <string>
#include "nn_sdk.h"

void* init_network(const char* model_path);
int uninit_network(void* qcontext);
std::tuple<cv::Mat, float, std::tuple<int, int>> preprocess(cv::Mat img, std::tuple<int, int> new_shape);
void* run_network(void* qcontext, std::vector<std::tuple<cv::Mat, float, std::tuple<int, int>>> input_tuples);

#endif
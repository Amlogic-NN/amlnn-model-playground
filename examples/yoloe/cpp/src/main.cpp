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


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "nn_sdk.h"
#include "model_loader.h"


#define HEIGHT 288
#define WIDTH 512
#define NUM_BOX 3024

struct PreprocessParam {
    float scale;
    int pad_x;
    int pad_y;
    int ori_w;
    int ori_h;
};


struct Box {
    int x1, y1, x2, y2;
    float conf;
};


struct PostprocessParam {
    float conf_thresh = 0.4f;
    float nms_thresh = 0.5f;
};


static int preprocess(const cv::Mat& src, cv::Mat& dst, PreprocessParam& param) {
    int h = 0;
    int w = 0;
    float scale = 0.0f;
    int nh = 0;
    int nw = 0;
    cv::Mat img_resized;
    h = src.rows;
    w = src.cols;
    scale = std::min(HEIGHT / static_cast<float>(h),
                        WIDTH / static_cast<float>(w));
    nh = static_cast<int>(h * scale);
    nw = static_cast<int>(w * scale);
    cv::resize(src, img_resized, cv::Size(nw, nh));

    int top = (HEIGHT - nh) / 2;
    int left = (WIDTH - nw) / 2;

    cv::Mat back(HEIGHT,WIDTH, CV_8UC3, cv::Scalar(114, 114, 114));
    img_resized.copyTo(back(cv::Rect(left, top, nw, nh)));
    back.convertTo(dst, CV_32F, 1.0 / 255.0);

    param.scale = scale;
    param.pad_x = left;
    param.pad_y = top;
    param.ori_w = w;
    param.ori_h = h;

    return 0;
}




static int postprocess(const float* nn_box_output, const float* nn_conf_output,
                            const PreprocessParam& pre_param,
                            const PostprocessParam& post_param,
                            std::vector<Box>& boxes) {
    std::vector<cv::Rect> rects;
    std::vector<float> confs;
    for (int i = 0; i < NUM_BOX; i++) {
        float conf = 1/(1 + std::exp(-nn_conf_output[i]));
        if (conf < post_param.conf_thresh) continue;

        float cx = nn_box_output[0 * NUM_BOX + i];
        float cy = nn_box_output[1 * NUM_BOX + i];
        float w = nn_box_output[2 * NUM_BOX + i];
        float h = nn_box_output[3 * NUM_BOX + i];

        float x1 = cx - w / 2;
        float y1 = cy - h / 2;

        int ori_x1 = static_cast<int>((x1 - pre_param.pad_x) / pre_param.scale);
        int ori_y1 = static_cast<int>((y1 - pre_param.pad_y) / pre_param.scale);
        int ori_w = static_cast<int>(w / pre_param.scale);
        int ori_h = static_cast<int>(h / pre_param.scale);

        ori_x1 = std::max(0, std::min(ori_x1, pre_param.ori_w - 1));
        ori_y1 = std::max(0, std::min(ori_y1, pre_param.ori_h - 1));
        ori_w = std::min(ori_w, pre_param.ori_w - ori_x1);
        ori_h = std::min(ori_h, pre_param.ori_h - ori_y1);
        if (ori_w <= 0 || ori_h <= 0) continue;

        confs.push_back(conf);
        rects.emplace_back(ori_x1, ori_y1, ori_w, ori_h);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(rects, confs, post_param.conf_thresh, post_param.nms_thresh, indices);
    for (auto i : indices) {
        auto& rect = rects[i];
        Box box;
        box.x1 = rect.x;
        box.y1 = rect.y;
        box.x2 = rect.x + rect.width;
        box.y2 = rect.y + rect.height;
        box.conf = confs[i];
        boxes.push_back(box);
    }

    return 0;
}


int main(int argc, char** argv) {
    std::string model_path = argv[1];
    std::string image_path = argv[2];

    void* context = init_network(model_path.c_str());
    if (!context) {
        std::cerr << "Failed to initialize network." << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image from " << image_path << std::endl;
        uninit_network(context);
        return -1;
    }

    cv::Mat img_rgb, processed_img;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    PreprocessParam pre_param;
    preprocess(img_rgb, processed_img, pre_param);

    nn_input inData;
    memset(&inData, 0, sizeof(nn_input));
    inData.input_type = BINARY_RAW_DATA;
    inData.input = processed_img.data;
    inData.input_index = 0;
    inData.size = WIDTH * HEIGHT * 3 * sizeof(float);

    if (aml_module_input_set(context, &inData) != 0) {
         std::cerr << "Failed to set input." << std::endl;
         uninit_network(context);
         return -1;
    }

    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.format = AML_OUTDATA_FLOAT32;
    
    nn_output* outdata = (nn_output*)aml_module_output_get(context, outconfig);
    if (!outdata) {
         std::cerr << "Failed to run network (get output)." << std::endl;
         uninit_network(context);
         return -1;
    }

    float* box_output = (float*)outdata->out[1].buf;
    float* conf_output = (float*)outdata->out[2].buf;

    PostprocessParam post_param;
    std::vector<Box> boxes;
    postprocess(box_output, conf_output, pre_param, post_param, boxes);

    std::cout << "Objects:\n";
    for (auto& box : boxes) {
        std::cout << "[ " << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << " ]\n"; 
    }
    std::cout << std::endl;
    
    uninit_network(context);

    return 0;
}



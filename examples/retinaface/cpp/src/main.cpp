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
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "nn_sdk.h"
#include "postprocess.h"

namespace fs = std::filesystem;

static void hwc_to_chw(const cv::Mat& src, float* dst) {
    int h = src.rows, w = src.cols;
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                dst[k * h * w + i * w + j] = src.at<cv::Vec3f>(i, j)[k];
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n";
        return 0;
    }

    aml_config cfg{};
    cfg.typeSize = sizeof(cfg);
    cfg.modelType = ADLA_LOADABLE;
    cfg.nbgType = NN_ADLA_FILE;
    cfg.path = argv[1];
    void* ctx = aml_module_create(&cfg);
    if (!ctx) {
        std::cerr << "Failed to create aml_module\n";
        return -1;
    }

    auto priors = generate_priors();
    size_t num_priors = priors.size();
    std::vector<float> chw_buffer(kInputW * kInputH * 3);

    const std::string out_dir = "retinaface_result";
    fs::create_directory(out_dir);

    std::vector<fs::path> image_paths;
    for (auto& it : fs::directory_iterator(argv[2])) {
        std::string ext = it.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
            image_paths.push_back(it.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    int total = image_paths.size();
    if (total == 0) {
        std::cout << "No images found in " << argv[2] << "\n";
        aml_module_destroy(ctx);
        return 0;
    }

    for (int i = 0; i < total; ++i) {
        const auto& path = image_paths[i];
        const std::string filename = path.filename().string();

        std::cout << "============================================================\n";
        std::cout << "Processing image " << (i + 1) << "/" << total << ": " << filename << "\n";
        std::cout << "============================================================\n";

        cv::Mat img = cv::imread(path.string());
        if (img.empty()) continue;

        float scale = std::min((float)kInputW / img.cols, (float)kInputH / img.rows);
        int nw = img.cols * scale, nh = img.rows * scale;
        int px = (kInputW - nw) / 2, py = (kInputH - nh) / 2;
        cv::Mat res, canvas = cv::Mat::zeros(kInputH, kInputW, CV_32FC3);
        cv::resize(img, res, {nw, nh});
        res.convertTo(res, CV_32FC3);
        res.copyTo(canvas(cv::Rect(px, py, nw, nh)));
        hwc_to_chw(canvas, chw_buffer.data());

        nn_input in{};
        in.typeSize = sizeof(in);
        in.input_type = BINARY_RAW_DATA;
        in.input = (unsigned char*)chw_buffer.data();
        in.size = chw_buffer.size() * 4;
        in.info.valid = 1;
        in.info.input_format = AML_INPUT_MODEL_NCHW;
        in.info.input_data_type = AML_INPUT_FP32;
        aml_module_input_set(ctx, &in);

        aml_output_config_t outcfg{};
        outcfg.typeSize = sizeof(outcfg);
        outcfg.format = AML_OUTDATA_FLOAT32;
        nn_output* out = (nn_output*)aml_module_output_get(ctx, outcfg);
        if (!out) continue;

        float *loc = nullptr, *conf = nullptr, *landm = nullptr;
        for (int j = 0; j < out->num; j++) {
            if (out->out[j].size == num_priors * 4 * 4) loc = (float*)out->out[j].buf;
            else if (out->out[j].size == num_priors * 2 * 4) conf = (float*)out->out[j].buf;
            else if (out->out[j].size == num_priors * 10 * 4) landm = (float*)out->out[j].buf;
        }
        if (!loc || !conf || !landm) continue;

        bool is_planar = (conf[0] > 2.0 || conf[1] > 2.0);
        std::vector<std::array<float, 4>> boxes;
        std::vector<std::array<float, 10>> lms;
        std::vector<float> scores_vec;

        for (size_t j = 0; j < num_priors; j++) {
            float sc = is_planar ? conf[num_priors + j] : conf[j * 2 + 1];
            if (sc > 0.5f) {
                boxes.push_back(decode_box(loc, j, num_priors, is_planar, priors[j]));
                lms.push_back(decode_landm(landm, j, num_priors, is_planar, priors[j]));
                scores_vec.push_back(sc);
            }
        }

        auto keep = nms(boxes, scores_vec, 0.4f);
        for (int k : keep) {
            auto& b = boxes[k];
            int x1 = (b[0] * kInputW - px) / scale, y1 = (b[1] * kInputH - py) / scale;
            int x2 = (b[2] * kInputW - px) / scale, y2 = (b[3] * kInputH - py) / scale;
            
            cv::rectangle(img, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);

            char score_text[16];
            std::snprintf(score_text, sizeof(score_text), "%.2f", scores_vec[k]);
            cv::putText(img, score_text, {x1, std::max(y1 - 5, 5)}, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1, cv::LINE_AA);

            auto& lm = lms[k];
            for (int j = 0; j < 5; j++) {
                int lx = (lm[2 * j] * kInputW - px) / scale;
                int ly = (lm[2 * j + 1] * kInputH - py) / scale;
                cv::circle(img, {lx, ly}, 2, {0, 0, 255}, -1); 
            }
        }

        std::string save_path = out_dir + "/" + filename;
        cv::imwrite(save_path, img);
        
        std::cout << "    Detected " << keep.size() << " faces\n";
        std::cout << "    Result saved to: " << save_path << "\n\n";
    }

    aml_module_destroy(ctx);
    return 0;
}
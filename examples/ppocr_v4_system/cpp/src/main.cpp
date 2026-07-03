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
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "nnsdk2.h"
#include "model_loader.h"
#include "postprocess.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <det_model.adla> <rec_model.adla> <image_dir> <dict_path>\n", argv[0]);
        return -1;
    }

    std::string det_model_path = argv[1];
    std::string rec_model_path = argv[2];
    std::string dict_path = argv[4];

    std::cout << "PPOCR Demo" << std::endl;

    std::string output_dir = "ppocr_result";
    fs::create_directory(output_dir);

    // ============================================
    // 1. Initialize Networks
    // ============================================
    void *det_context = nullptr;
    void *rec_context = nullptr;

    if (init_network(det_model_path, det_context) != AMLNN_SUCCESS)
    {
        fprintf(stderr, "Failed to init DET network.\n");
        return -1;
    }
    if (init_network(rec_model_path, rec_context) != AMLNN_SUCCESS)
    {
        fprintf(stderr, "Failed to init REC network.\n");
        uninit_network(det_context);
        return -1;
    }

    amlnn_input_output_num det_io, rec_io;
    amlnn_query(det_context, AMLNN_QUERY_IN_OUT_NUM, &det_io, sizeof(det_io));
    amlnn_query(rec_context, AMLNN_QUERY_IN_OUT_NUM, &rec_io, sizeof(rec_io));

    if (det_io.n_output != 1 || rec_io.n_output != 1)
    {
        std::cerr << "Warning: Expected 1 output each. Det model has "
                  << det_io.n_output << " outputs and Rec model has "
                  << rec_io.n_output << " outputs" << std::endl;
    }

    // Prepare REC input quantization parameters (Int16)
    amlnn_tensor_attr rec_in_attr = query_input_attr(rec_context, 0);
    amlnn_tensor_attr det_in_attr = query_input_attr(det_context, 0);

    // Query Det Output Attributes for shapes
    amlnn_tensor_attr det_out_attr = query_output_attr(det_context, 0);
    std::vector<int> det_out_shape = get_tensor_shape(det_out_attr);

    // Query Rec Output Attributes for shapes
    amlnn_tensor_attr rec_out_attr = query_output_attr(rec_context, 0);
    std::vector<int> rec_out_shape = get_tensor_shape(rec_out_attr);

    // Prepare Output vectors
    std::vector<amlnn_output> det_outData(det_io.n_output);
    std::vector<amlnn_output> rec_outData(rec_io.n_output);

    // Load dictionary
    std::vector<std::string> char_dict = load_dict(dict_path);
    if (char_dict.empty())
    {
        fprintf(stderr, "Failed to load dictionary.\n");
        return -1;
    }

    for (auto &it : fs::directory_iterator(argv[3]))
    {
        if (!it.is_regular_file())
            continue;

        // 2. Load Image
        cv::Mat img = cv::imread(it.path().string());
        if (img.empty())
        {
            std::cerr << "Failed to load image from " << it.path().string() << std::endl;
            continue;
        }

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        // ============================================
        // 2. DET PIPELINE (Input: Uint8)
        // ============================================
        auto [float_image, scale] = preprocess_det(img, DET_MODEL_WIDTH, DET_MODEL_HEIGHT);
        std::vector<uint8_t> prepared_data = prepare_input_tensor(float_image, det_in_attr);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Must strictly be 640 * 640 * 3 = 1,228,800
        size_t det_input_size = prepared_data.size();

        if (!run_network(det_context, prepared_data.data(), det_input_size, det_outData))
        {
            fprintf(stderr, "DET Inference failed\n");
            uninit_network(det_context);
            uninit_network(rec_context);
            return -1;
        }

        auto end_det_time = std::chrono::high_resolution_clock::now();

        if (det_outData.empty() || det_outData[0].buf == nullptr)
        {
            std::cerr << "Invalid output data" << std::endl;
            return -1;
        }

        std::vector<Object> det_results = postprocess_det((float *)det_outData[0].buf, det_out_shape, img, BOX_SCORE_THRESH, BOX_THRESH, scale);

        printf("[RESULT] Detected %zu objects.\n", det_results.size());

        std::cout << "------------------------------------------------------------" << std::endl
                  << std::endl;
        // ============================================
        // 3. REC PIPELINE (Input: Int16)
        // ============================================
        std::chrono::duration<double, std::milli> total_rec_time{0};

        for (size_t i = 0; i < det_results.size(); ++i)
        {
            // Find bounding rectangle of the polygon to crop the text line
            cv::Rect rect = cv::boundingRect(det_results[i].box);

            // Ensure bounds are safe inside the image
            rect.x = std::max(0, rect.x);
            rect.y = std::max(0, rect.y);
            rect.width = std::min(img.cols - rect.x, rect.width);
            rect.height = std::min(img.rows - rect.y, rect.height);

            if (rect.width <= 0 || rect.height <= 0)
                continue;

            cv::Mat rec_crop = img(rect);

            // // Aspect ratio safety check
            // float crop_ratio = (float)rec_crop.cols / (float)rec_crop.rows;
            // int target_width = int(REC_MODEL_HEIGHT * crop_ratio);
            // if (target_width > REC_MODEL_WIDTH * 1.2f)
            // {
            //     printf("[WARNING] Box %zu text is too long. Skipping...\n", i);
            //     continue;
            // }

            // Preprocess crop for REC (Creates Float32 normalized buffer)
            cv::Mat rec_float_img = preprocess_rec(rec_crop, REC_MODEL_WIDTH, REC_MODEL_HEIGHT);
            std::vector<uint8_t> prepared_data = prepare_input_tensor(rec_float_img, rec_in_attr);

            // Run REC Inference
            auto start_rec_time = std::chrono::high_resolution_clock::now();

            size_t rec_input_size = prepared_data.size();
            if (!run_network(rec_context, prepared_data.data(), rec_input_size, rec_outData))
            {
                fprintf(stderr, "REC Inference failed on box %zu\n", i);
                continue;
            }

            if (rec_outData.empty() || rec_outData[0].buf == nullptr)
            {
                continue;
            }

            auto end_time = std::chrono::high_resolution_clock::now();

            // Postprocess REC
            std::string rec_res = postprocess_rec((float *)rec_outData[0].buf, rec_out_shape, char_dict);

            printf("[RESULT] Box %zu - Text: %s \n", i, rec_res.c_str());

            total_rec_time += end_time - start_rec_time;
        }

        std::chrono::duration<double, std::milli> total_det_time = end_det_time - start_time;
        std::chrono::duration<double, std::milli> avg_rec_time = total_rec_time / det_results.size();
        std::chrono::duration<double, std::milli> total_time = total_det_time + total_rec_time;

        printf("\n[TIMING] Det Inference Time: %.2f ms\n", total_det_time.count());
        printf("\n[TIMING] Avg Rec Inference Time: %.2f ms X %zu times\n", avg_rec_time.count(), det_results.size());
        printf("\n[TIMING] Total OCR Inference Time: %.2f ms\n", total_time.count());

        // ============================================
        // 4. Draw results
        // ============================================
        fs::path out_path = fs::path(output_dir) / it.path().filename();
        cv::Mat res = draw_ocr_results(img, det_results);
        cv::imwrite(out_path.string(), res);
        printf("Saved output to %s\n", out_path.c_str());
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    // ============================================
    // 5. Cleanup
    // ============================================
    uninit_network(det_context);
    uninit_network(rec_context);

    return 0;
}
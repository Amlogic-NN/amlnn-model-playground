/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
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
#include <chrono>
#include <tuple>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

const int TOP_K = 5;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir> <labels.txt>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::cout << "EVA-02 Demo" << std::endl;
    fs::create_directory("eva02_result");

    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    amlnn_tensor_attr input_attr = query_input_attr(context, 0);

    std::vector<int> input_shape = get_tensor_shape(input_attr);

    std::cout << "Input shape: [";
    for (size_t i = 0; i < input_shape.size(); ++i)
    {
        std::cout << input_shape[i];
        if (i + 1 < input_shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    int input_height = input_shape[0];
    int input_width = input_shape[1];

    std::vector<amlnn_tensor_attr> out_attrs;
    std::vector<std::vector<int>> out_shapes;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_attrs.push_back(query_output_attr(context, i));
        out_shapes.push_back(get_tensor_shape(out_attrs[i]));
    }

    std::vector<amlnn_output> outData(io_num.n_output);
    std::vector<std::string> class_names = load_class_names(argv[3]);

    for (auto &it : fs::directory_iterator(argv[2]))
    {
        if (!it.is_regular_file())
            continue;

        cv::Mat img = cv::imread(it.path().string());
        if (img.empty())
            continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        cv::Mat preprocessed = preprocess(img, std::make_tuple(input_height, input_width));
        std::vector<uint8_t> prepared_data = prepare_input_tensor(preprocessed, input_attr);

        if (prepared_data.empty())
        {
            std::cerr << "Failed to prepare input tensor." << std::endl;
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(context, prepared_data.data(), prepared_data.size(), outData))
        {
            std::cerr << "Failed to run network" << std::endl;
            return -1;
        }

        if (outData.empty())
            return -1;

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        std::vector<float *> out_ptrs;
        for (int i = 0; i < io_num.n_output; ++i)
        {
            out_ptrs.push_back((float *)outData[i].buf);
        }

        std::vector<ClassificationResult> results = postprocess(out_ptrs, out_shapes, class_names, TOP_K);

        std::cout << "Top " << results.size() << " results:" << std::endl;
        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << "  " << i + 1 << ". " << results[i].class_name << " (" << std::fixed
                      << std::setprecision(4) << results[i].score << ")" << std::endl;
        }
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    uninit_network(context);

    return 0;
}

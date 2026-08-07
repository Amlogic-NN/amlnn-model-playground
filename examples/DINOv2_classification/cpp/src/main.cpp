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

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << " <backbone.adla> <classifier.adla> <image_dir> <labels.txt> [topk]\n";
        return 0;
    }

    std::string backbone_model_path = argv[1];
    std::string classifier_model_path = argv[2];
    std::string image_dir = argv[3];
    std::string labels_path = argv[4];
    int topk = argc > 5 ? std::stoi(argv[5]) : 5;

    std::cout << "DINOv2 Linear Classification Demo" << std::endl;

    void *backbone_context = nullptr;
    void *classifier_context = nullptr;

    int ret = init_network(backbone_model_path, backbone_context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize backbone network. Error: " << ret << std::endl;
        return -1;
    }

    ret = init_network(classifier_model_path, classifier_context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize classifier network. Error: " << ret << std::endl;
        uninit_network(backbone_context);
        return -1;
    }

    amlnn_input_output_num backbone_io_num;
    amlnn_query(backbone_context, AMLNN_QUERY_IN_OUT_NUM, &backbone_io_num, sizeof(backbone_io_num));

    amlnn_input_output_num classifier_io_num;
    amlnn_query(classifier_context, AMLNN_QUERY_IN_OUT_NUM, &classifier_io_num, sizeof(classifier_io_num));

    amlnn_tensor_attr backbone_input_attr = query_input_attr(backbone_context, 0);
    std::vector<int> backbone_input_shape = get_tensor_shape(backbone_input_attr);

    std::cout << "Backbone input shape: [";
    for (size_t i = 0; i < backbone_input_shape.size(); ++i)
    {
        std::cout << backbone_input_shape[i];
        if (i + 1 < backbone_input_shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    int input_height = backbone_input_shape[0];
    int input_width = backbone_input_shape[1];

    std::vector<amlnn_tensor_attr> backbone_output_attrs;
    std::vector<std::vector<int>> backbone_output_shapes;
    for (int i = 0; i < backbone_io_num.n_output; ++i)
    {
        backbone_output_attrs.push_back(query_output_attr(backbone_context, i));
        backbone_output_shapes.push_back(get_tensor_shape(backbone_output_attrs[i]));

        std::cout << "Backbone output " << i << " shape: [";
        for (size_t j = 0; j < backbone_output_shapes[i].size(); ++j)
        {
            std::cout << backbone_output_shapes[i][j];
            if (j + 1 < backbone_output_shapes[i].size())
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    amlnn_tensor_attr classifier_input_attr = query_input_attr(classifier_context, 0);
    amlnn_tensor_attr classifier_output_attr = query_output_attr(classifier_context, 0);
    std::vector<int> classifier_input_shape = get_tensor_shape(classifier_input_attr);
    std::vector<int> classifier_output_shape = get_tensor_shape(classifier_output_attr);

    std::cout << "Classifier input shape: [";
    for (size_t i = 0; i < classifier_input_shape.size(); ++i)
    {
        std::cout << classifier_input_shape[i];
        if (i + 1 < classifier_input_shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Classifier output shape: [";
    for (size_t i = 0; i < classifier_output_shape.size(); ++i)
    {
        std::cout << classifier_output_shape[i];
        if (i + 1 < classifier_output_shape.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::vector<std::string> class_names = load_class_names(labels_path);
    std::vector<amlnn_output> backbone_outData(backbone_io_num.n_output);
    std::vector<amlnn_output> classifier_outData(classifier_io_num.n_output);

    for (auto &it : fs::directory_iterator(image_dir))
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
        std::vector<uint8_t> prepared_backbone_input = prepare_input_tensor(preprocessed, backbone_input_attr);

        if (prepared_backbone_input.empty())
        {
            std::cerr << "Failed to prepare backbone input tensor." << std::endl;
            continue;
        }

        auto backbone_start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(backbone_context, prepared_backbone_input.data(), prepared_backbone_input.size(), backbone_outData))
        {
            std::cerr << "Failed to run backbone network" << std::endl;
            return -1;
        }

        auto backbone_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> backbone_inference_time = backbone_end_time - backbone_start_time;
        std::cout << "Backbone inference time: " << backbone_inference_time.count() << " ms" << std::endl;

        if (backbone_outData.empty())
            return -1;

        std::vector<float> concat_features = concatenate_backbone_outputs(
            backbone_outData,
            backbone_output_shapes,
            classifier_input_attr.n_elems);

        if (concat_features.empty())
        {
            std::cerr << "Failed to concatenate backbone outputs." << std::endl;
            continue;
        }

        std::vector<uint8_t> prepared_classifier_input = prepare_feature_tensor(concat_features, classifier_input_attr);

        if (prepared_classifier_input.empty())
        {
            std::cerr << "Failed to prepare classifier input tensor." << std::endl;
            continue;
        }

        auto classifier_start_time = std::chrono::high_resolution_clock::now();

        if (!run_network(classifier_context, prepared_classifier_input.data(), prepared_classifier_input.size(), classifier_outData))
        {
            std::cerr << "Failed to run classifier network" << std::endl;
            return -1;
        }

        auto classifier_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> classifier_inference_time = classifier_end_time - classifier_start_time;
        std::cout << "Classifier inference time: " << classifier_inference_time.count() << " ms" << std::endl;

        if (classifier_outData.empty())
            return -1;

        float *classifier_output = reinterpret_cast<float *>(classifier_outData[0].buf);
        std::vector<std::pair<int, float>> results = postprocess(classifier_output, classifier_output_attr.n_elems, topk);

        std::cout << "Results:" << std::endl;
        for (size_t i = 0; i < results.size(); ++i)
        {
            int class_id = results[i].first;
            float confidence = results[i].second;
            std::string class_name = class_id < static_cast<int>(class_names.size())
                ? class_names[class_id]
                : "class_" + std::to_string(class_id);

            std::cout << "  " << i + 1 << ". " << class_name
                      << " (" << std::fixed << std::setprecision(4)
                      << confidence << ")" << std::endl;
        }
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    uninit_network(backbone_context);
    uninit_network(classifier_context);

    return 0;
}
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
#include <chrono>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

namespace fs = std::filesystem;

struct DecoderInputIndices
{
    int embeddings = -1;
    int point_coords = -1;
    int point_labels = -1;
    int mask_input = -1;
    int has_mask_input = -1;
};

bool run_network_multi(void *context, const std::vector<std::vector<uint8_t>> &input_buffers,
                       std::vector<amlnn_output> &outputs)
{
    std::vector<amlnn_input> inputs(input_buffers.size());

    for (size_t i = 0; i < input_buffers.size(); ++i)
    {
        memset(&inputs[i], 0, sizeof(amlnn_input));
        inputs[i].index = i;
        inputs[i].buf = const_cast<uint8_t *>(input_buffers[i].data());
        inputs[i].size = input_buffers[i].size();
    }

    if (amlnn_inputs_set(context, static_cast<uint32_t>(inputs.size()), inputs.data()) != AMLNN_SUCCESS)
        return false;

    if (amlnn_run(context, nullptr) != AMLNN_SUCCESS)
        return false;

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        memset(&outputs[i], 0, sizeof(amlnn_output));
        outputs[i].is_float = 1;
        outputs[i].index = i;
    }

    return amlnn_outputs_get(context, static_cast<uint32_t>(outputs.size()), outputs.data()) == AMLNN_SUCCESS;
}

DecoderInputIndices discover_decoder_inputs(const std::vector<amlnn_tensor_attr> &attrs, int embedding_elements)
{
    DecoderInputIndices indices;

    for (size_t i = 0; i < attrs.size(); ++i)
    {
        int elements = get_tensor_element_count(attrs[i]);

        if (elements == embedding_elements)
            indices.embeddings = i;
        else if (elements == 4)
            indices.point_coords = i;
        else if (elements == 2)
            indices.point_labels = i;
        else if (elements == 1)
            indices.has_mask_input = i;
    }

    int largest_remaining = 0;

    for (size_t i = 0; i < attrs.size(); ++i)
    {
        if (static_cast<int>(i) == indices.embeddings ||
            static_cast<int>(i) == indices.point_coords ||
            static_cast<int>(i) == indices.point_labels ||
            static_cast<int>(i) == indices.has_mask_input)
            continue;

        int elements = get_tensor_element_count(attrs[i]);

        if (elements > largest_remaining)
        {
            largest_remaining = elements;
            indices.mask_input = i;
        }
    }

    return indices;
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cout << "Usage: " << argv[0] << " <encoder.adla> <decoder.adla> <image> <point|box> <values>\n";
        std::cout << "Point: \"x,y,label\" or \"x1,y1,l1;x2,y2,l2\"\n";
        std::cout << "Box:   \"x1,y1,x2,y2\"\n";
        return 0;
    }

    std::string encoder_path = argv[1];
    std::string decoder_path = argv[2];
    std::string image_path = argv[3];
    std::string prompt_type = argv[4];
    std::string prompt_values = argv[5];

    std::cout << "MobileSAM Demo" << std::endl;
    fs::create_directory("mobilesam_result");

    void *encoder_context = nullptr;
    void *decoder_context = nullptr;

    int ret = init_network(encoder_path, encoder_context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize encoder. Error: " << ret << std::endl;
        return -1;
    }

    ret = init_network(decoder_path, decoder_context);
    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize decoder. Error: " << ret << std::endl;
        uninit_network(encoder_context);
        return -1;
    }

    amlnn_input_output_num encoder_io;
    amlnn_input_output_num decoder_io;
    amlnn_query(encoder_context, AMLNN_QUERY_IN_OUT_NUM, &encoder_io, sizeof(encoder_io));
    amlnn_query(decoder_context, AMLNN_QUERY_IN_OUT_NUM, &decoder_io, sizeof(decoder_io));

    if (encoder_io.n_input != 1 || encoder_io.n_output != 1)
    {
        std::cerr << "Expected encoder to have 1 input and 1 output." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    if (decoder_io.n_input != 5 || decoder_io.n_output < 2)
    {
        std::cerr << "Expected decoder to have 5 inputs and at least 2 outputs." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    amlnn_tensor_attr encoder_input_attr = query_input_attr(encoder_context, 0);
    amlnn_tensor_attr encoder_output_attr = query_output_attr(encoder_context, 0);
    std::vector<int> encoder_input_shape = get_tensor_shape(encoder_input_attr);

    if (encoder_input_shape.size() < 3)
    {
        std::cerr << "Invalid encoder input shape." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    int input_height = encoder_input_shape[0];
    int input_width = encoder_input_shape[1];
    std::cout << "Encoder input shape: [" << input_height << ", " << input_width << ", 3]" << std::endl;

    std::vector<amlnn_tensor_attr> decoder_input_attrs;
    std::vector<amlnn_tensor_attr> decoder_output_attrs;

    for (int i = 0; i < decoder_io.n_input; ++i)
        decoder_input_attrs.push_back(query_input_attr(decoder_context, i));

    for (int i = 0; i < decoder_io.n_output; ++i)
        decoder_output_attrs.push_back(query_output_attr(decoder_context, i));

    int embedding_elements = get_tensor_element_count(encoder_output_attr);
    DecoderInputIndices decoder_inputs = discover_decoder_inputs(decoder_input_attrs, embedding_elements);

    if (decoder_inputs.embeddings < 0 || decoder_inputs.point_coords < 0 ||
        decoder_inputs.point_labels < 0 || decoder_inputs.mask_input < 0 ||
        decoder_inputs.has_mask_input < 0)
    {
        std::cerr << "Failed to identify decoder inputs." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    std::vector<std::pair<int, int>> decoder_outputs;

    for (int i = 0; i < decoder_io.n_output; ++i)
        decoder_outputs.push_back({get_tensor_element_count(decoder_output_attrs[i]), i});

    std::sort(decoder_outputs.begin(), decoder_outputs.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    int mask_output_index = decoder_outputs[0].second;
    int score_output_index = decoder_outputs[1].second;

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    ImageMeta meta;
    Prompt prompt;
    std::vector<float> point_coords;
    std::vector<float> point_labels;

    cv::Mat preprocessed = preprocess(image, std::make_tuple(input_height, input_width), meta);

    if (!build_prompt(prompt_type, prompt_values, meta, point_coords, point_labels, prompt))
    {
        std::cerr << "Invalid prompt." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    std::vector<uint8_t> encoder_input = prepare_input_tensor(preprocessed, encoder_input_attr);
    if (encoder_input.empty())
    {
        std::cerr << "Failed to prepare encoder input." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    std::vector<amlnn_output> encoder_outputs(encoder_io.n_output);

    auto encoder_start = std::chrono::high_resolution_clock::now();

    if (!run_network(encoder_context, encoder_input.data(), encoder_input.size(), encoder_outputs))
    {
        std::cerr << "Failed to run encoder." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    auto encoder_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> encoder_time = encoder_end - encoder_start;
    std::cout << "Encoder inference time: " << encoder_time.count() << " ms" << std::endl;

    std::vector<std::vector<uint8_t>> decoder_input_buffers(decoder_io.n_input);

    // Image embeddings
    decoder_input_buffers[decoder_inputs.embeddings] =
        prepare_tensor(static_cast<float *>(encoder_outputs[0].buf), embedding_elements,
                       decoder_input_attrs[decoder_inputs.embeddings]);

    // Point coordinates
    decoder_input_buffers[decoder_inputs.point_coords] =
        prepare_tensor(point_coords.data(), point_coords.size(),
                       decoder_input_attrs[decoder_inputs.point_coords]);

    // Point labels
    decoder_input_buffers[decoder_inputs.point_labels] =
        prepare_tensor(point_labels.data(), point_labels.size(),
                       decoder_input_attrs[decoder_inputs.point_labels]);

    // Previous mask
    int mask_input_elements = get_tensor_element_count(decoder_input_attrs[decoder_inputs.mask_input]);
    std::vector<float> mask_input(mask_input_elements, 0.0f);
    decoder_input_buffers[decoder_inputs.mask_input] =
        prepare_tensor(mask_input.data(), mask_input.size(),
                       decoder_input_attrs[decoder_inputs.mask_input]);

    // Whether a previous mask is provided
    float has_mask_input = 0.0f;
    decoder_input_buffers[decoder_inputs.has_mask_input] =
        prepare_tensor(&has_mask_input, 1, decoder_input_attrs[decoder_inputs.has_mask_input]);

    for (const auto &buffer : decoder_input_buffers)
    {
        if (buffer.empty())
        {
            std::cerr << "Failed to prepare decoder inputs." << std::endl;
            uninit_network(decoder_context);
            uninit_network(encoder_context);
            return -1;
        }
    }

    std::vector<amlnn_output> decoder_output_data(decoder_io.n_output);

    auto decoder_start = std::chrono::high_resolution_clock::now();

    if (!run_network_multi(decoder_context, decoder_input_buffers, decoder_output_data))
    {
        std::cerr << "Failed to run decoder." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    auto decoder_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> decoder_time = decoder_end - decoder_start;
    std::cout << "Decoder inference time: " << decoder_time.count() << " ms" << std::endl;

    MaskResult result = postprocess(
        static_cast<float *>(decoder_output_data[mask_output_index].buf),
        decoder_output_attrs[mask_output_index],
        static_cast<float *>(decoder_output_data[score_output_index].buf),
        get_tensor_element_count(decoder_output_attrs[score_output_index]),
        meta
    );

    if (result.mask.empty())
    {
        std::cerr << "Failed to postprocess mask." << std::endl;
        uninit_network(decoder_context);
        uninit_network(encoder_context);
        return -1;
    }

    cv::Mat result_image = draw_result(image, result.mask, prompt);
    std::string filename = fs::path(image_path).stem().string();
    std::string result_path = "mobilesam_result/" + filename + "_result.png";
    std::string mask_path = "mobilesam_result/" + filename + "_mask.png";

    cv::imwrite(result_path, result_image);
    cv::imwrite(mask_path, result.mask);

    std::cout << "Mask index: " << result.index << std::endl;
    std::cout << "Mask score: " << result.score << std::endl;
    std::cout << "Result saved to: " << result_path << std::endl;
    std::cout << "Mask saved to: " << mask_path << std::endl;

    uninit_network(decoder_context);
    uninit_network(encoder_context);
    return 0;
}
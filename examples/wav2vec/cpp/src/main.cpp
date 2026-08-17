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
#include <cstdint>
#include <cctype>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

const int SAMPLE_RATE = 16000;
const int TARGET_SAMPLES = 320000;
const int OVERLAP_SECONDS = 2;
const int OUTPUT_CHANNELS = 32;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <audio_dir>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::cout << "Wav2Vec2 Demo" << std::endl;

    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    if (io_num.n_input != 1 || io_num.n_output != 1)
    {
        std::cerr << "Expected 1 input and 1 output, got " << io_num.n_input
                  << " input(s) and " << io_num.n_output << " output(s)" << std::endl;
        uninit_network(context);
        return -1;
    }

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

    int target_samples = get_element_count(input_shape);
    if (target_samples != TARGET_SAMPLES)
    {
        std::cerr << "Expected " << TARGET_SAMPLES << " input samples, got " << target_samples << std::endl;
        uninit_network(context);
        return -1;
    }

    std::vector<amlnn_tensor_attr> out_attrs;
    std::vector<std::vector<int>> out_shapes;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_attrs.push_back(query_output_attr(context, i));
        out_shapes.push_back(get_tensor_shape(out_attrs[i]));
    }

    if (out_shapes[0].empty())
    {
        std::cerr << "Invalid output shape" << std::endl;
        uninit_network(context);
        return -1;
    }

    int output_channels = out_shapes[0].back();
    int output_elements = get_element_count(out_shapes[0]);
    int output_steps = output_elements / output_channels;

    if (output_channels != OUTPUT_CHANNELS)
    {
        std::cerr << "Expected " << OUTPUT_CHANNELS << " output channels, got " << output_channels << std::endl;
        uninit_network(context);
        return -1;
    }

    int overlap_samples = OVERLAP_SECONDS * SAMPLE_RATE;
    int overlap_output_steps = static_cast<int>(std::round(
        static_cast<double>(overlap_samples) * output_steps / target_samples));

    std::vector<amlnn_output> outData(io_num.n_output);
    std::vector<fs::path> audio_files;

    for (auto &it : fs::directory_iterator(argv[2]))
    {
        if (!it.is_regular_file())
            continue;

        std::string extension = it.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });

        if (extension == ".wav")
            audio_files.push_back(it.path());
    }

    std::sort(audio_files.begin(), audio_files.end());

    if (audio_files.empty())
    {
        std::cout << "No WAV files found." << std::endl;
        uninit_network(context);
        return 0;
    }

    for (size_t file_index = 0; file_index < audio_files.size(); ++file_index)
    {
        const fs::path &audio_path = audio_files[file_index];

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing [" << file_index + 1 << "/" << audio_files.size()
                  << "]: " << audio_path.filename().string() << std::endl;
        std::cout << "============================================================" << std::endl;

        std::vector<float> waveform;
        uint32_t original_sample_rate = 0;

        if (!load_wav(audio_path.string(), waveform, original_sample_rate))
        {
            std::cerr << "Error processing " << audio_path.filename().string()
                      << ": Failed to load WAV file" << std::endl;
            continue;
        }

        waveform = resample_audio(waveform, static_cast<int>(original_sample_rate), SAMPLE_RATE);
        if (waveform.empty())
        {
            std::cerr << "Error processing " << audio_path.filename().string()
                      << ": Audio file contains no samples" << std::endl;
            continue;
        }

        std::vector<AudioSegment> segments = preprocess_audio(waveform, target_samples, overlap_samples);
        if (segments.empty())
        {
            std::cerr << "Error processing " << audio_path.filename().string()
                      << ": Failed to preprocess audio" << std::endl;
            continue;
        }

        std::cout << "Segments: " << segments.size() << std::endl;
        std::vector<float> combined_logits;
        bool inference_failed = false;

        for (size_t segment_index = 0; segment_index < segments.size(); ++segment_index)
        {
            std::cout << "Processing segment [" << segment_index + 1 << "/" << segments.size() << "]..." << std::endl;

            std::vector<uint8_t> prepared_data = prepare_input_tensor(segments[segment_index].waveform, input_attr);

            if (prepared_data.empty())
            {
                std::cerr << "Failed to prepare input tensor." << std::endl;
                inference_failed = true;
                break;
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            if (!run_network(context, prepared_data.data(), prepared_data.size(), outData))
            {
                std::cerr << "Failed to run network" << std::endl;
                inference_failed = true;
                break;
            }

            if (outData.empty())
            {
                inference_failed = true;
                break;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
            std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

            std::vector<float *> out_ptrs;
            for (int i = 0; i < io_num.n_output; ++i)
            {
                out_ptrs.push_back((float *)outData[i].buf);
            }

            int valid_output_steps = static_cast<int>(std::round(
                static_cast<double>(segments[segment_index].real_samples) * output_steps / target_samples));
            valid_output_steps = std::min(std::max(valid_output_steps, 1), output_steps);

            int keep_start = segment_index == 0 ? 0 : overlap_output_steps;
            int keep_end = segment_index + 1 == segments.size()
                               ? valid_output_steps
                               : output_steps - overlap_output_steps;

            if (!append_retained_logits(out_ptrs[0], output_steps, output_channels,
                                        keep_start, keep_end, combined_logits))
            {
                std::cerr << "Invalid retained output range [" << keep_start << ":" << keep_end
                          << "] for segment " << segment_index + 1 << std::endl;
                inference_failed = true;
                break;
            }
        }

        if (inference_failed)
        {
            std::cerr << "Error processing " << audio_path.filename().string() << std::endl;
            continue;
        }

        std::string transcription = postprocess(combined_logits, output_channels);
        std::cout << "Transcription: " << transcription << std::endl;
    }

    std::cout << "============================================================" << std::endl
              << std::endl;

    uninit_network(context);

    return 0;
}
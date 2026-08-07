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
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include "postprocess.h"
#include "nnsdk2.h"
#include "model_loader.h"

namespace fs = std::filesystem;
const float MAX_DURATION = 15.0f;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <model.adla> <audio_dir> <yamnet_class_map.csv>\n";
        return 0;
    }

    std::string model_path = argv[1];
    std::string audio_dir = argv[2];
    std::string labels_path = argv[3];

    std::cout << "YAMNet NPU Audio Classification Demo (FP16)" << std::endl;

    // Load Labels
    auto class_names = load_class_names(labels_path);

    // 1. Initialize Network
    void *context = nullptr;
    int ret = init_network(model_path, context);

    if (ret != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize network. Error: " << ret << std::endl;
        return -1;
    }

    // Query Inputs and Outputs
    amlnn_input_output_num io_num;
    amlnn_query(context, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    amlnn_tensor_attr output_attr = query_output_attr(context, 0);

    std::vector<amlnn_output> outData(1);

    for (auto &it : fs::directory_iterator(audio_dir))
    {
        if (!it.is_regular_file()) continue;

        // Filter WAV only
        std::string ext = it.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".wav") continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing audio: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        // 2. Load Audio
        uint32_t original_sr = 0;
        std::vector<float> audio_data;
        if (!load_wav(it.path().string(), audio_data, original_sr)) {
            std::cerr << "Failed to load/parse WAV: " << it.path().filename().string() << std::endl;
            continue;
        }

        // Resample if necessary
        if (original_sr != 16000) {
            std::cout << "  Resampling from " << original_sr << "Hz to 16000Hz..." << std::endl;
            audio_data = resample_audio(audio_data, original_sr, 16000);
        }

        // 3. Preprocess
        auto frames = preprocess_audio(audio_data, 16000, MAX_DURATION);
        if (frames.empty()) continue;

        std::vector<float> file_predictions(output_attr.n_elems, 0.0f);
        double total_inference_time = 0.0;

        // 4. Run Network Frame by Frame
        for (const auto& frame : frames) {
            size_t input_size = frame.size() * sizeof(float);

            auto start_time = std::chrono::high_resolution_clock::now();

            if (!run_network(context, (void*)frame.data(), input_size, outData)) {
                std::cerr << "Failed to run network on frame" << std::endl;
                break;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
            total_inference_time += inference_time.count();

            float* output_data_ptr = (float*)outData[0].buf;
            for (int j = 0; j < output_attr.n_elems; ++j) {
                file_predictions[j] += output_data_ptr[j];
            }
        }

        std::cout << "Total Inference time (all frames): " << total_inference_time << " ms" << std::endl;

        // 5. Average Out and Calculate Top 5
        for (float& score : file_predictions) {
            score /= frames.size();
        }

        auto top5 = get_top_k(file_predictions, class_names, 5);
        for (int rank = 0; rank < top5.size(); ++rank) {
            std::cout << "  " << (rank + 1) << ". "
                      << std::left << std::setw(30) << top5[rank].class_name
                      << " (" << std::fixed << std::setprecision(4) << top5[rank].score << ")" << std::endl;
        }
    }

    std::cout << "============================================================" << std::endl << std::endl;

    // 6. Cleanup
    uninit_network(context);

    return 0;
}
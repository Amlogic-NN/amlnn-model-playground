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

#include <stdio.h>
#include <time.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "post_process_whisper.h"
#include "pre_process_whisper.h"
#include "whisper_assets.h"
#include "whisper_invoke.h"

#define BILLION 1000000000ULL
#define OVERLAP_SECONDS 2

namespace fs = std::filesystem;

struct Get_Times
{
    uint64_t init_total_time = 0;
    uint64_t preProcess_total_time = 0;
    uint64_t invoke_total_time = 0;
    uint64_t total_time = 0;
    std::vector<uint64_t> total_time_group;
};

static uint64_t get_time_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_nsec) + static_cast<uint64_t>(ts.tv_sec) * BILLION;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        printf("Usage: %s <encoder.adla> <decoder.adla> <audio.wav> <data_bin_dir>\n", argv[0]);
        return -1;
    }

    const char *model_path_encoder = argv[1];
    const char *model_path_decoder = argv[2];
    const std::string audio_path = argv[3];
    const fs::path data_bin_dir = fs::absolute(argv[4]);
    const fs::path mel_filter_path = data_bin_dir / "data.bin";
    const fs::path tokenizer_path = data_bin_dir / "tokenizer_info.bin";

    if (!fs::is_regular_file(mel_filter_path))
    {
        std::cerr << "Mel filter file not found: " << mel_filter_path << std::endl;
        return -1;
    }

    if (!fs::is_regular_file(tokenizer_path))
    {
        std::cerr << "Tokenizer file not found: " << tokenizer_path << std::endl;
        return -1;
    }

    set_whisper_mel_filter_path(mel_filter_path.string());

    Get_Times encoder_time, decoder_time, whisper_time;
    void *context_enc = nullptr;
    void *context_dec = nullptr;
    int result = -1;

    const uint64_t init_start_time = get_time_count();
    context_enc = init_network_file(model_path_encoder);
    context_dec = init_network_file(model_path_decoder);
    whisper_time.init_total_time = (get_time_count() - init_start_time) / 1000000ULL;

    if (context_enc == nullptr || context_dec == nullptr)
    {
        printf("Network initialization failed.\n");
        destroy_network(context_enc);
        destroy_network(context_dec);
        return -1;
    }

    WhisperModelInfo model_info;
    if (!query_whisper_model_info(context_enc, context_dec, model_info))
    {
        destroy_network(context_enc);
        destroy_network(context_dec);
        return -1;
    }

    print_tensor_info("Encoder input", model_info.encoder_input);
    print_tensor_info("Encoder output", model_info.encoder_output);
    print_tensor_info("Decoder input 0", model_info.decoder_ids_input);
    print_tensor_info("Decoder input 1", model_info.decoder_hidden_input);
    print_tensor_info("Decoder output", model_info.decoder_output);

    const std::vector<int> encoder_shape = get_tensor_shape(model_info.encoder_input);
    if (encoder_shape.size() != 2 || encoder_shape[0] != WHISPER_N_MELS)
    {
        printf("Expected encoder input shape [1, %d, frames], got ", WHISPER_N_MELS);
        print_shape(encoder_shape);
        destroy_network(context_enc);
        destroy_network(context_dec);
        return -1;
    }

    const int n_mel = encoder_shape[0];
    const int n_frames = encoder_shape[1];

    whisper_vocab vocab = read_token_info(tokenizer_path.string());
    if (vocab.id_to_token.empty())
    {
        printf("Failed to load tokenizer information.\n");
        destroy_network(context_enc);
        destroy_network(context_dec);
        return -1;
    }

    std::cout << "Mel filter file: " << mel_filter_path << std::endl;
    std::cout << "Tokenizer file: " << tokenizer_path << std::endl;

    if (getenv("GET_TIME"))
    {
        std::cout << "init_whisper_total time : " << whisper_time.init_total_time << "ms" << std::endl;
    }

    const uint64_t preprocess_start_time = get_time_count();
    std::vector<std::vector<float>> encoder_inputs = do_pre_process(
        audio_path,
        n_mel,
        n_frames,
        OVERLAP_SECONDS
    );
    whisper_time.preProcess_total_time = (get_time_count() - preprocess_start_time) / 1000000ULL;

    if (encoder_inputs.empty())
    {
        destroy_network(context_enc);
        destroy_network(context_dec);
        return -1;
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "Processing audio: " << audio_path << std::endl;
    std::cout << "Segments: " << encoder_inputs.size() << std::endl;
    std::cout << "============================================================" << std::endl;

    std::vector<std::string> segment_transcriptions;
    segment_transcriptions.reserve(encoder_inputs.size());

    bool inference_ok = true;

    for (size_t segment_index = 0; segment_index < encoder_inputs.size(); ++segment_index)
    {
        std::cout << "Processing segment [" << segment_index + 1 << "/" << encoder_inputs.size() << "]..." << std::endl;

        std::vector<float> encoder_output_data;
        const uint64_t encoder_start_time = get_time_count();

        if (!run_network_encoder_process(
                context_enc,
                encoder_inputs[segment_index],
                model_info.encoder_input,
                model_info.encoder_output,
                encoder_output_data))
        {
            inference_ok = false;
            break;
        }

        const uint64_t encoder_elapsed_ms = (get_time_count() - encoder_start_time) / 1000000ULL;
        encoder_time.total_time_group.push_back(encoder_elapsed_ms);

        std::string segment_text;
        if (!run_network_decoder(
                context_dec,
                encoder_output_data,
                model_info,
                vocab,
                segment_text,
                &decoder_time.total_time_group))
        {
            inference_ok = false;
            break;
        }

        segment_transcriptions.push_back(segment_text);
        std::cout << "Segment transcription: " << segment_text << std::endl;
    }

    if (inference_ok)
    {
        const std::string final_transcription = merge_transcriptions(segment_transcriptions);

        std::cout << "============================================================" << std::endl;
        std::cout << "Audio Text:" << std::endl;
        std::cout << final_transcription << std::endl;
        std::cout << "============================================================" << std::endl;
        result = 0;
    }

    if (getenv("GET_OUTPUTS_SIZE"))
    {
        std::cout << "==================================" << std::endl;
        std::cout << "WHISPER_OUTPUTS_SIZE : " << decoder_time.total_time_group.size() << std::endl;
    }

    if (getenv("GET_TIME"))
    {
        encoder_time.invoke_total_time = std::accumulate(
            encoder_time.total_time_group.begin(),
            encoder_time.total_time_group.end(),
            uint64_t{0}
        );
        decoder_time.invoke_total_time = std::accumulate(
            decoder_time.total_time_group.begin(),
            decoder_time.total_time_group.end(),
            uint64_t{0}
        );

        std::cout << "pre-process time             : " << whisper_time.preProcess_total_time << "ms" << std::endl;

        for (size_t i = 0; i < encoder_time.total_time_group.size(); ++i)
        {
            std::cout << "encoder inference time[" << i << "]  : " << encoder_time.total_time_group[i] << "ms" << std::endl;
        }

        std::cout << "============================================================" << std::endl;

        for (size_t i = 0; i < decoder_time.total_time_group.size(); ++i)
        {
            std::cout << "decoder inference time[" << i << "]  : " << decoder_time.total_time_group[i] << "ms" << std::endl;
        }

        whisper_time.total_time = whisper_time.preProcess_total_time +
                                  encoder_time.invoke_total_time +
                                  decoder_time.invoke_total_time;

        std::cout << "============================================================" << std::endl;

        if (!encoder_time.total_time_group.empty())
        {
            std::cout << "model->whisper encoder avg : "
                      << encoder_time.invoke_total_time / encoder_time.total_time_group.size()
                      << "ms" << std::endl;
        }

        if (!decoder_time.total_time_group.empty())
        {
            std::cout << "model->whisper decoder avg : "
                      << decoder_time.invoke_total_time / decoder_time.total_time_group.size()
                      << "ms" << std::endl;
        }

        std::cout << "model->whisper total time  : " << whisper_time.total_time << "ms" << std::endl;
    }

    destroy_network(context_enc);
    destroy_network(context_dec);

    return result;
}
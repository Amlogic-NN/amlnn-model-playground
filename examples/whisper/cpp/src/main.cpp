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
#include <iostream>

#include "whisper_invoke.h"
#include "nnsdk2.h"

#define BILLION 1000000000
#define GET_INFERENCE_TIME (1)
#define WHISPER_DECODER_INPUTS 48

struct Get_Times
{
    uint64_t init_start_time = 0, init_end_time = 0, init_total_time = 0;
    uint64_t preProcess_start_time = 0, preProcess_end_time = 0, preProcess_total_time = 0;
    uint64_t invoke_start_time = 0, invoke_end_time = 0, invoke_total_time = 0;
    uint64_t total_time = 0;
    std::vector<uint64_t> total_time_group;
};

static uint64_t get_time_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
}

int main(int argc, char **argv)
{
    Get_Times encoder_time, decoder_time, whisper_time;
    Input_Decoder decoder_inputs_data;
    std::vector<float> encoder_input_data;
    std::vector<float> encoder_output_data;

    int64_t input_1_data[] = {50257, 50362}; /* init token, for tiny_en or base_en */
    int input_1_data_size = sizeof(input_1_data) / sizeof(input_1_data[0]);

    int ret = 0;
    if (argc < 4)
    {
        printf("Usage: %s <encoder.adla> <decoder.adla> <audio.wav>\n", argv[0]);
        return -1;
    }

    char *model_path_encoder = argv[1];
    char *model_path_decoder = argv[2];
    void *context_enc = NULL;
    void *context_dec = NULL;

    whisper_time.init_start_time = get_time_count();
    context_enc = init_network_file(model_path_encoder);
    context_dec = init_network_file(model_path_decoder);
    whisper_time.init_end_time = get_time_count();

    whisper_time.init_total_time = (whisper_time.init_end_time - whisper_time.init_start_time) / 1000000;

    if (context_enc == NULL || context_dec == NULL)
    {
        printf("Network initialization failed.\n");
        return -1;
    }

    if (getenv("GET_TIME"))
    {
        std::cout << "init_whisper_total time : " << whisper_time.init_total_time << "ms" << std::endl;
    }

    std::string provided_audio = argv[3];
    bool first_run = true;

    std::string input_str;
    bool is_finish = false;
    std::string out_text = "start";

    input_str = provided_audio;
    first_run = false;

    decoder_inputs_data.input_1_size = WHISPER_DECODER_INPUTS;
    decoder_inputs_data.input_1 = new int64_t[decoder_inputs_data.input_1_size];
    std::copy(input_1_data, input_1_data + input_1_data_size, decoder_inputs_data.input_1);

    // Fill remaining with 0
    std::fill(decoder_inputs_data.input_1 + input_1_data_size,
              decoder_inputs_data.input_1 + decoder_inputs_data.input_1_size, 0);

    whisper_time.preProcess_start_time = get_time_count();

    encoder_input_data = do_pre_process(input_str);

    whisper_time.preProcess_end_time = get_time_count();

    encoder_output_data = run_network_encoder_process(context_enc, encoder_input_data);
    encoder_time.invoke_end_time = get_time_count();

    decoder_inputs_data.input_0_size = encoder_output_data.size();
    decoder_inputs_data.input_0 = new float[decoder_inputs_data.input_0_size];
    std::copy(encoder_output_data.begin(), encoder_output_data.end(), decoder_inputs_data.input_0);

    whisper_time.preProcess_total_time = (whisper_time.preProcess_end_time - whisper_time.preProcess_start_time) / 1000000;
    encoder_time.invoke_total_time = (encoder_time.invoke_end_time - whisper_time.preProcess_end_time) / 1000000;

    std::cout << "============================================================" << std::endl;
    printf("\nAudio Text:\n");

    while (!is_finish)
    {
        decoder_time.invoke_start_time = get_time_count();
        out_text = run_network_decoder(context_dec, &decoder_inputs_data);
        decoder_time.invoke_end_time = get_time_count();
        is_finish = is_finish_end();
        decoder_time.total_time_group.push_back((decoder_time.invoke_end_time - decoder_time.invoke_start_time) / 1000000);
        std::cout << out_text << std::flush;
    }
    printf("\n");
    std::cout << "============================================================" << std::endl;

    if (getenv("GET_OUTPUTS_SIZE"))
    {
        std::cout << "==================================" << std::endl;
        std::cout << "WHISPER_OUTPUTS_SIZE : " << decoder_time.total_time_group.size() << std::endl;
    }

    if (getenv("GET_TIME"))
    {
        uint64_t total_time_whisper;
        for (int i = 0; i < decoder_time.total_time_group.size(); i++)
        {
            if (i < 1)
            {
                total_time_whisper = whisper_time.preProcess_total_time + encoder_time.invoke_total_time;
                whisper_time.total_time = total_time_whisper;
                std::cout << "pre-process time             : " << whisper_time.preProcess_total_time << "ms" << std::endl;
                std::cout << "encoder_inference_total time : " << encoder_time.invoke_total_time << "ms" << std::endl;
                std::cout << "============================================================" << std::endl;
            }
            decoder_time.invoke_total_time += decoder_time.total_time_group[i];
            std::cout << "decoder inference time[" << i << "]  : " << decoder_time.total_time_group[i] << "ms" << std::endl;
        }

        whisper_time.total_time += decoder_time.invoke_total_time;
        std::cout << "============================================================" << std::endl;
        std::cout << "model->whisper decoder avg : " << decoder_time.invoke_total_time / decoder_time.total_time_group.size() << "ms" << std::endl;
        std::cout << "model->whisper total time  : " << whisper_time.total_time<< "ms" << std::endl;
        whisper_time.total_time = decoder_time.invoke_total_time = 0;
    }
    encoder_time.total_time_group.clear();
    std::cout << "============================================================" << std::endl;

    if (decoder_inputs_data.input_0 != nullptr)
    {
        delete[] decoder_inputs_data.input_0;
        decoder_inputs_data.input_0 = nullptr;
    }

    if (decoder_inputs_data.input_1 != nullptr)
    {
        delete[] decoder_inputs_data.input_1;
        decoder_inputs_data.input_1 = nullptr;
    }

    destroy_network(context_enc);
    destroy_network(context_dec);

    return 0;
}
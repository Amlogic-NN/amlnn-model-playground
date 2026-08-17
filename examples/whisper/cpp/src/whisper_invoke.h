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

#ifndef WHISPER_INVOKE_H
#define WHISPER_INVOKE_H

#include <cstdint>
#include <string>
#include <vector>

#include "nnsdk2.h"
#include "whisper.h"

struct WhisperModelInfo
{
    amlnn_tensor_attr encoder_input{};
    amlnn_tensor_attr encoder_output{};
    amlnn_tensor_attr decoder_ids_input{};
    amlnn_tensor_attr decoder_hidden_input{};
    amlnn_tensor_attr decoder_output{};

    int decoder_length = 0;
    int vocab_size = 0;
};

void *init_network_file(const char *model_path);
bool query_whisper_model_info(void *encoder_context, void *decoder_context, WhisperModelInfo &model_info);

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);
void print_shape(const std::vector<int> &shape);
void print_tensor_info(const char *label, const amlnn_tensor_attr &attr);

bool run_network_encoder_process(
    void *context,
    const std::vector<float> &input_features,
    const amlnn_tensor_attr &input_attr,
    const amlnn_tensor_attr &output_attr,
    std::vector<float> &encoder_output
);

bool run_network_decoder(
    void *context,
    const std::vector<float> &encoder_output,
    const WhisperModelInfo &model_info,
    const whisper_vocab &vocab,
    std::string &transcription,
    std::vector<uint64_t> *decoder_inference_times_ms
);

int destroy_network(void *context);

#endif // WHISPER_INVOKE_H
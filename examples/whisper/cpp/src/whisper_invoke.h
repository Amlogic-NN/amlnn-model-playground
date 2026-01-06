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

#include <string>
#include <vector>
#include <map>
#include "nn_sdk.h"

struct Input_Decoder {
    float * input_0;
    int input_0_size;
    int64_t * input_1;
    int input_1_size;
};

void* init_network_file(const char *model_path);
std::vector<float> do_pre_process(std::string fname_inp);
std::vector<float> run_network_encoder_process(void *qcontext, std::vector<float> input_ids);
std::string run_network_decoder(void *qcontext_sec, Input_Decoder* input_data);
bool is_finish_end();
int destroy_network(void *qcontext);

#endif // WHISPER_INVOKE_H

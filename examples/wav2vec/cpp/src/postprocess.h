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

#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cstdint>
#include <string>
#include <vector>
#include "nnsdk2.h"

struct AudioSegment
{
    std::vector<float> waveform;
    int real_samples;
};

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr);
int get_element_count(const std::vector<int> &shape);
bool load_wav(const std::string &path, std::vector<float> &waveform, uint32_t &out_sr);
std::vector<float> resample_audio(const std::vector<float> &input, int original_sr, int target_sr);
std::vector<AudioSegment> preprocess_audio(const std::vector<float> &waveform,
                                           int target_samples, int overlap_samples);
std::vector<uint8_t> prepare_input_tensor(const std::vector<float> &waveform,
                                          const amlnn_tensor_attr &attr);
bool append_retained_logits(const float *logits, int output_steps, int output_channels,
                            int keep_start, int keep_end, std::vector<float> &combined_logits);
std::string postprocess(const std::vector<float> &combined_logits, int output_channels);

#endif // POSTPROCESS_H
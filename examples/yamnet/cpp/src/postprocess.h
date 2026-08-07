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

#include <vector>
#include <string>
#include <map>
#include <cstdint>

struct ClassificationResult {
    int class_id;
    std::string class_name;
    float score;
};

// Load class names from the YAMNet CSV file
std::map<int, std::string> load_class_names(const std::string& csv_path);

// Robust WAV loader that handles odd-byte chunks and multiple bit-depths
bool load_wav(const std::string& path, std::vector<float>& waveform, uint32_t& out_sr);

// Linear interpolation resampler
std::vector<float> resample_audio(const std::vector<float>& input, int original_sr, int target_sr);

// Replicates librosa waveform chunking, scaling, and zero-padding
std::vector<std::vector<float>> preprocess_audio(std::vector<float> waveform, int sr, float max_duration);

// Sort and fetch Top-K classifications
std::vector<ClassificationResult> get_top_k(const std::vector<float>& scores, const std::map<int, std::string>& class_names, int k);

#endif // POSTPROCESS_H
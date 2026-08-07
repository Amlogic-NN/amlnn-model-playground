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

#include "postprocess.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>

std::map<int, std::string> load_class_names(const std::string& csv_path) {
    std::map<int, std::string> class_names;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << csv_path << ". Fallback to generic IDs.\n";
        return class_names;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        size_t first_comma = line.find(',');
        if (first_comma == std::string::npos) continue;
        int id = std::stoi(line.substr(0, first_comma));

        size_t second_comma = line.find(',', first_comma + 1);
        if (second_comma == std::string::npos) continue;

        std::string name = line.substr(second_comma + 1);

        // Strip hidden carriage returns (\r) and newlines (\n) that break terminal printing
        name.erase(std::remove(name.begin(), name.end(), '\r'), name.end());
        name.erase(std::remove(name.begin(), name.end(), '\n'), name.end());

        if (!name.empty() && name.front() == '"' && name.back() == '"') {
            name = name.substr(1, name.size() - 2);
        }
        class_names[id] = name;
    }
    return class_names;
}

bool load_wav(const std::string& path, std::vector<float>& waveform, uint32_t& out_sr) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    char chunk[4];
    if (!file.read(chunk, 4) || std::strncmp(chunk, "RIFF", 4) != 0) return false;
    file.seekg(4, std::ios::cur);
    if (!file.read(chunk, 4) || std::strncmp(chunk, "WAVE", 4) != 0) return false;

    uint16_t format = 1, channels = 1, bits = 16;
    uint32_t sr = 16000, chunk_size = 0;
    bool data_found = false;

    // Dynamically parse chunks to handle padding correctly
    while (file.read(chunk, 4)) {
        if (!file.read(reinterpret_cast<char*>(&chunk_size), 4)) break;
        uint32_t pad = chunk_size % 2; // RIFF specifies chunks must be even-byte aligned

        if (std::strncmp(chunk, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char*>(&format), 2);
            file.read(reinterpret_cast<char*>(&channels), 2);
            file.read(reinterpret_cast<char*>(&sr), 4);
            file.seekg(6, std::ios::cur); // skip byterate and block_align
            file.read(reinterpret_cast<char*>(&bits), 2);

            int remaining = chunk_size - 16;
            if (remaining > 0) file.seekg(remaining, std::ios::cur);
            if (pad) file.seekg(1, std::ios::cur);
        } else if (std::strncmp(chunk, "data", 4) == 0) {
            data_found = true;
            break; // Stop exactly at data start
        } else {
            file.seekg(chunk_size + pad, std::ios::cur); // Skip unknown chunks properly
        }
    }

    if (!data_found || channels == 0 || bits == 0) return false;
    out_sr = sr;

    int bytes_per_sample = bits / 8;
    int num_samples = chunk_size / bytes_per_sample;
    int num_frames = num_samples / channels;
    waveform.resize(num_frames, 0.0f);

    std::vector<uint8_t> raw_data(chunk_size);
    file.read(reinterpret_cast<char*>(raw_data.data()), chunk_size);

    for (int i = 0; i < num_frames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            int idx = (i * channels + c) * bytes_per_sample;
            float val = 0.0f;

            if (bits == 16) {
                int16_t pcm = *reinterpret_cast<int16_t*>(&raw_data[idx]);
                val = pcm / 32768.0f;
            } else if (bits == 24) {
                int32_t pcm = raw_data[idx] | (raw_data[idx+1] << 8) | (raw_data[idx+2] << 16);
                if (pcm & 0x800000) pcm |= 0xFF000000; // Sign extend
                val = pcm / 8388608.0f;
            } else if (bits == 32 && format == 3) { // IEEE Float
                val = *reinterpret_cast<float*>(&raw_data[idx]);
            } else if (bits == 32 && format == 1) { // 32-bit Integer PCM
                int32_t pcm = *reinterpret_cast<int32_t*>(&raw_data[idx]);
                val = pcm / 2147483648.0f;
            } else if (bits == 8) { // 8-bit unsigned PCM
                val = (raw_data[idx] - 128) / 128.0f;
            }
            sum += val;
        }
        waveform[i] = sum / channels; // Downmix stereo to mono
    }
    return true;
}

std::vector<float> resample_audio(const std::vector<float>& input, int original_sr, int target_sr) {
    if (original_sr == target_sr || input.empty()) return input;

    int target_length = static_cast<int>(input.size() * static_cast<double>(target_sr) / original_sr);
    std::vector<float> output(target_length);
    double ratio = static_cast<double>(original_sr) / target_sr;

    for (int i = 0; i < target_length; ++i) {
        double src_idx = i * ratio;
        int idx1 = static_cast<int>(src_idx);
        int idx2 = std::min(idx1 + 1, static_cast<int>(input.size() - 1));
        double frac = src_idx - idx1;
        output[i] = static_cast<float>((1.0 - frac) * input[idx1] + frac * input[idx2]);
    }
    return output;
}

std::vector<std::vector<float>> preprocess_audio(std::vector<float> waveform, int sr, float max_duration) {
    float max_val = 0.0f;
    for (float v : waveform) {
        if (std::abs(v) > max_val) max_val = std::abs(v);
    }
    if (max_val > 1.0f) {
        for (float& v : waveform) v /= max_val;
    }

    size_t max_samples = static_cast<size_t>(max_duration * sr);
    if (waveform.size() > max_samples) {
        waveform.resize(max_samples);
    }

    size_t window_size = 15360;
    size_t step_size = 7680;

    if (waveform.size() < window_size) {
        waveform.resize(window_size, 0.0f);
    }

    std::vector<std::vector<float>> frames;
    for (size_t i = 0; i + window_size <= waveform.size(); i += step_size) {
        std::vector<float> frame(waveform.begin() + i, waveform.begin() + i + window_size);
        frames.push_back(frame);
    }
    return frames;
}

std::vector<ClassificationResult> get_top_k(const std::vector<float>& scores, const std::map<int, std::string>& class_names, int k) {
    std::vector<std::pair<float, int>> indexed_scores;
    for (int i = 0; i < scores.size(); ++i) {
        indexed_scores.push_back({scores[i], i});
    }

    std::sort(indexed_scores.rbegin(), indexed_scores.rend());

    std::vector<ClassificationResult> results;
    int limit = std::min<int>(k, indexed_scores.size());
    for (int i = 0; i < limit; ++i) {
        int class_idx = indexed_scores[i].second;
        std::string name = "Class_" + std::to_string(class_idx);
        if (class_names.count(class_idx)) {
            name = class_names.at(class_idx);
        }
        results.push_back({class_idx, name, indexed_scores[i].first});
    }
    return results;
}
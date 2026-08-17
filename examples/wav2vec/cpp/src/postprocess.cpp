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
#include <limits>
#include <utility>

const char *TOKENIZER_DICT[32] = {
    "<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I",
    "H", "S", "R", "D", "L", "U", "M", "W", "C", "F", "G",
    "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"};

std::vector<int> get_tensor_shape(const amlnn_tensor_attr &attr)
{
    std::vector<int> shape;
    for (int i = 0; i < attr.n_dims; ++i)
    {
        if (attr.dims[i] > 1)
        {
            shape.push_back(attr.dims[i]);
        }
    }
    return shape;
}

int get_element_count(const std::vector<int> &shape)
{
    int count = 1;
    for (int dim : shape)
        count *= dim;
    return count;
}

bool load_wav(const std::string &path, std::vector<float> &waveform, uint32_t &out_sr)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return false;

    char chunk[4];
    if (!file.read(chunk, 4) || std::strncmp(chunk, "RIFF", 4) != 0)
        return false;

    file.seekg(4, std::ios::cur);
    if (!file.read(chunk, 4) || std::strncmp(chunk, "WAVE", 4) != 0)
        return false;

    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint32_t sample_rate = 0;
    uint32_t data_size = 0;
    bool format_found = false;
    bool data_found = false;

    while (file.read(chunk, 4))
    {
        uint32_t chunk_size = 0;
        if (!file.read(reinterpret_cast<char *>(&chunk_size), sizeof(chunk_size)))
            break;

        uint32_t pad = chunk_size % 2;

        if (std::strncmp(chunk, "fmt ", 4) == 0)
        {
            if (chunk_size < 16)
                return false;

            uint32_t byte_rate = 0;
            uint16_t block_align = 0;
            file.read(reinterpret_cast<char *>(&format), sizeof(format));
            file.read(reinterpret_cast<char *>(&channels), sizeof(channels));
            file.read(reinterpret_cast<char *>(&sample_rate), sizeof(sample_rate));
            file.read(reinterpret_cast<char *>(&byte_rate), sizeof(byte_rate));
            file.read(reinterpret_cast<char *>(&block_align), sizeof(block_align));
            file.read(reinterpret_cast<char *>(&bits), sizeof(bits));

            if (!file)
                return false;

            if (chunk_size > 16)
                file.seekg(chunk_size - 16, std::ios::cur);
            if (pad)
                file.seekg(1, std::ios::cur);

            format_found = true;
        }
        else if (std::strncmp(chunk, "data", 4) == 0)
        {
            data_size = chunk_size;
            data_found = true;
            break;
        }
        else
        {
            file.seekg(chunk_size + pad, std::ios::cur);
        }
    }

    if (!format_found || !data_found || channels == 0 || sample_rate == 0)
        return false;

    if (format != 1 && format != 3)
        return false;

    if ((format == 1 && bits != 8 && bits != 16 && bits != 24 && bits != 32) ||
        (format == 3 && bits != 32))
        return false;

    int bytes_per_sample = bits / 8;
    int frame_size = bytes_per_sample * channels;
    if (frame_size <= 0)
        return false;

    int num_frames = static_cast<int>(data_size / frame_size);
    if (num_frames <= 0)
        return false;

    std::vector<uint8_t> raw_data(data_size);
    if (!file.read(reinterpret_cast<char *>(raw_data.data()), data_size))
        return false;

    waveform.resize(num_frames, 0.0f);

    for (int i = 0; i < num_frames; ++i)
    {
        float sum = 0.0f;

        for (int c = 0; c < channels; ++c)
        {
            size_t index = static_cast<size_t>(i * channels + c) * bytes_per_sample;
            float value = 0.0f;

            if (bits == 8)
            {
                value = (static_cast<int>(raw_data[index]) - 128) / 128.0f;
            }
            else if (bits == 16)
            {
                int16_t pcm = 0;
                std::memcpy(&pcm, raw_data.data() + index, sizeof(pcm));
                value = pcm / 32768.0f;
            }
            else if (bits == 24)
            {
                int32_t pcm = static_cast<int32_t>(raw_data[index]) |
                              (static_cast<int32_t>(raw_data[index + 1]) << 8) |
                              (static_cast<int32_t>(raw_data[index + 2]) << 16);
                if (pcm & 0x800000)
                    pcm |= static_cast<int32_t>(0xFF000000);
                value = pcm / 8388608.0f;
            }
            else if (format == 3)
            {
                std::memcpy(&value, raw_data.data() + index, sizeof(value));
            }
            else
            {
                int32_t pcm = 0;
                std::memcpy(&pcm, raw_data.data() + index, sizeof(pcm));
                value = static_cast<float>(pcm / 2147483648.0);
            }

            sum += value;
        }

        waveform[i] = sum / channels;
    }

    out_sr = sample_rate;
    return true;
}

std::vector<float> resample_audio(const std::vector<float> &input, int original_sr, int target_sr)
{
    if (original_sr == target_sr || input.empty())
        return input;

    int target_length = static_cast<int>(input.size() * static_cast<double>(target_sr) / original_sr);
    if (target_length <= 0)
        return {};

    std::vector<float> output(target_length);
    double ratio = static_cast<double>(original_sr) / target_sr;

    for (int i = 0; i < target_length; ++i)
    {
        double source_index = i * ratio;
        int index1 = static_cast<int>(source_index);
        int index2 = std::min(index1 + 1, static_cast<int>(input.size() - 1));
        double fraction = source_index - index1;
        output[i] = static_cast<float>((1.0 - fraction) * input[index1] + fraction * input[index2]);
    }

    return output;
}

std::vector<AudioSegment> preprocess_audio(const std::vector<float> &waveform,
                                           int target_samples, int overlap_samples)
{
    if (waveform.empty())
        return {};

    int step_samples = target_samples - 2 * overlap_samples;
    if (step_samples <= 0)
    {
        std::cerr << "The left and right overlap must total less than the model input length" << std::endl;
        return {};
    }

    std::vector<AudioSegment> segments;
    size_t start = 0;

    while (true)
    {
        int real_samples = static_cast<int>(std::min(
            static_cast<size_t>(target_samples), waveform.size() - start));

        std::vector<float> segment(
            waveform.begin() + start,
            waveform.begin() + start + real_samples);

        double mean = 0.0;
        for (float value : segment)
            mean += value;
        mean /= real_samples;

        double variance = 0.0;
        for (float value : segment)
        {
            double difference = value - mean;
            variance += difference * difference;
        }
        variance /= real_samples;

        double denominator = std::sqrt(variance + 1e-7);
        for (float &value : segment)
            value = static_cast<float>((value - mean) / denominator);

        if (real_samples < target_samples)
            segment.resize(target_samples, 0.0f);

        segments.push_back({std::move(segment), real_samples});

        if (start + target_samples >= waveform.size())
            break;

        start += step_samples;
    }

    return segments;
}

std::vector<uint8_t> prepare_input_tensor(const std::vector<float> &waveform,
                                          const amlnn_tensor_attr &attr)
{
    std::vector<uint8_t> tensor_data;

    if (waveform.empty())
    {
        std::cerr << "prepare_input_tensor: Invalid input waveform" << std::endl;
        return tensor_data;
    }

    int total_elements = waveform.size();
    const float *src_ptr = waveform.data();

    if (attr.type == AMLNN_TENSOR_FLOAT32)
    {
        tensor_data.resize(total_elements * sizeof(float));
        std::memcpy(tensor_data.data(), waveform.data(), tensor_data.size());
    }
    else if (attr.type == AMLNN_TENSOR_INT16)
    {
        tensor_data.resize(total_elements * sizeof(int16_t));
        int16_t *dst_ptr = reinterpret_cast<int16_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_INT8)
    {
        tensor_data.resize(total_elements * sizeof(int8_t));
        int8_t *dst_ptr = reinterpret_cast<int8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
        }
    }
    else if (attr.type == AMLNN_TENSOR_UINT8)
    {
        tensor_data.resize(total_elements * sizeof(uint8_t));
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(tensor_data.data());
        for (int i = 0; i < total_elements; ++i)
        {
            float val = std::round(src_ptr[i] / attr.scale) + attr.zp;
            dst_ptr[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
        }
    }
    else
    {
        std::cerr << "prepare_input_tensor: Unsupported tensor type " << attr.type << std::endl;
    }

    return tensor_data;
}

bool append_retained_logits(const float *logits, int output_steps, int output_channels,
                            int keep_start, int keep_end, std::vector<float> &combined_logits)
{
    if (logits == nullptr || keep_start < 0 || keep_end > output_steps || keep_end <= keep_start)
        return false;

    const float *begin = logits + keep_start * output_channels;
    const float *end = logits + keep_end * output_channels;
    combined_logits.insert(combined_logits.end(), begin, end);
    return true;
}

std::string postprocess(const std::vector<float> &combined_logits, int output_channels)
{
    if (combined_logits.empty() || output_channels <= 0 ||
        combined_logits.size() % output_channels != 0)
        return "";

    int output_steps = combined_logits.size() / output_channels;
    std::vector<int> predicted_ids(output_steps);

    for (int step = 0; step < output_steps; ++step)
    {
        const float *step_logits = combined_logits.data() + step * output_channels;
        float max_value = -std::numeric_limits<float>::infinity();
        int token_id = 0;

        for (int channel = 0; channel < output_channels; ++channel)
        {
            if (step_logits[channel] > max_value)
            {
                max_value = step_logits[channel];
                token_id = channel;
            }
        }

        predicted_ids[step] = token_id;
    }

    std::string transcription;
    int previous_token = -1;

    for (int token_id : predicted_ids)
    {
        if (token_id == previous_token)
            continue;

        previous_token = token_id;

        if (token_id <= 3 || token_id >= 32)
            continue;

        if (token_id == 4)
            transcription += " ";
        else
            transcription += TOKENIZER_DICT[token_id];
    }

    size_t first = transcription.find_first_not_of(' ');
    if (first == std::string::npos)
        return "";

    size_t last = transcription.find_last_not_of(' ');
    return transcription.substr(first, last - first + 1);
}

/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
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

#include "wav_reader.h"

#include <cstring>
#include <fstream>

namespace {

struct WavHeader {
    char riff[4];
    uint32_t chunk_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

bool ReadExact(std::ifstream &ifs, void *buf, size_t size) {
    return static_cast<bool>(ifs.read(reinterpret_cast<char *>(buf),
                                      static_cast<std::streamsize>(size)));
}

}  // namespace

bool ReadWavPcm16(const std::string &path, WavData *out, std::string *error) {
    if (!out) {
        if (error) {
            *error = "output is null";
        }
        return false;
    }

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (error) {
            *error = "failed to open wav file";
        }
        return false;
    }

    WavHeader header{};
    if (!ReadExact(ifs, &header, sizeof(header))) {
        if (error) {
            *error = "invalid wav header";
        }
        return false;
    }

    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        if (error) {
            *error = "not a RIFF/WAVE file";
        }
        return false;
    }

    if (header.audio_format != 1 || header.bits_per_sample != 16) {
        if (error) {
            *error = "only PCM16 wav is supported";
        }
        return false;
    }

    char chunk_id[4];
    uint32_t chunk_size = 0;
    bool found_data = false;

    while (ifs.read(chunk_id, 4)) {
        if (!ReadExact(ifs, &chunk_size, sizeof(chunk_size))) {
            break;
        }

        if (std::strncmp(chunk_id, "data", 4) == 0) {
            found_data = true;
            break;
        }

        ifs.seekg(chunk_size, std::ios::cur);
    }

    if (!found_data || chunk_size == 0) {
        if (error) {
            *error = "wav data chunk not found";
        }
        return false;
    }

    const size_t num_samples_total = chunk_size / sizeof(int16_t);
    std::vector<int16_t> interleaved(num_samples_total);
    if (!ReadExact(ifs, interleaved.data(), chunk_size)) {
        if (error) {
            *error = "failed to read wav pcm data";
        }
        return false;
    }

    out->sample_rate = static_cast<int>(header.sample_rate);
    out->channels = header.num_channels;

    if (header.num_channels == 1) {
        out->samples = std::move(interleaved);
        return true;
    }

    const size_t frames = num_samples_total / header.num_channels;
    out->samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        int32_t sum = 0;
        for (uint16_t c = 0; c < header.num_channels; ++c) {
            sum += interleaved[i * header.num_channels + c];
        }
        out->samples[i] = static_cast<int16_t>(sum / header.num_channels);
    }

    return true;
}

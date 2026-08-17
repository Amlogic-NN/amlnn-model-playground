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

#include "pre_process_whisper.h"

#include <algorithm>
#include <cstdio>
#include <limits>

#include "pre_post_common.h"

std::vector<std::vector<float>> do_pre_process(
    const std::string &fname_inp,
    int n_mel,
    int n_frames,
    int overlap_seconds)
{
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;

    if (!read_wav(fname_inp, pcmf32, pcmf32s, false))
    {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
        return {};
    }

    if (pcmf32.empty())
    {
        fprintf(stderr, "error: WAV file contains no audio samples: '%s'\n", fname_inp.c_str());
        return {};
    }

    if (n_mel != WHISPER_N_MELS || n_frames <= 0)
    {
        fprintf(stderr, "error: unsupported encoder feature shape [%d, %d]\n", n_mel, n_frames);
        return {};
    }

    const int64_t target_samples_64 = static_cast<int64_t>(n_frames) * WHISPER_HOP_LENGTH;
    const int64_t overlap_samples_64 = static_cast<int64_t>(overlap_seconds) * WHISPER_SAMPLE_RATE;

    if (target_samples_64 <= 0 || target_samples_64 > std::numeric_limits<int>::max())
    {
        fprintf(stderr, "error: invalid model audio window size\n");
        return {};
    }

    if (overlap_samples_64 < 0 || overlap_samples_64 >= target_samples_64)
    {
        fprintf(stderr, "error: overlap must be shorter than the model audio window\n");
        return {};
    }

    const size_t target_samples = static_cast<size_t>(target_samples_64);
    const size_t overlap_samples = static_cast<size_t>(overlap_samples_64);
    const size_t step_samples = target_samples - overlap_samples;

    std::vector<std::vector<float>> input_segments;
    size_t start = 0;

    while (true)
    {
        const size_t end = std::min(start + target_samples, pcmf32.size());
        std::vector<float> segment(pcmf32.begin() + start, pcmf32.begin() + end);

        // The mel implementation applies a 200-sample reflective prefix.
        // Zero-extend extremely short clips so that prefix construction remains valid.
        const size_t minimum_samples = WHISPER_N_FFT / 2 + 1;
        if (segment.size() < minimum_samples)
        {
            segment.resize(minimum_samples, 0.0f);
        }

        whisper_context ctx{};
        whisper_state state{};

        if (whisper_pcm_to_mel_with_state(
                &ctx,
                &state,
                segment.data(),
                static_cast<int>(segment.size()),
                8) != 0)
        {
            fprintf(stderr, "error: failed to compute log mel spectrogram\n");
            return {};
        }

        if (state.mel.n_mel != n_mel || state.mel.n_len < n_frames)
        {
            fprintf(
                stderr,
                "error: mel output shape [%d, %d] cannot fill encoder input [%d, %d]\n",
                state.mel.n_mel,
                state.mel.n_len,
                n_mel,
                n_frames
            );
            return {};
        }

        std::vector<float> input_data(static_cast<size_t>(n_mel) * n_frames);

        for (int mel_index = 0; mel_index < n_mel; ++mel_index)
        {
            const float *source = state.mel.data.data() +
                                  static_cast<size_t>(mel_index) * state.mel.n_len;
            float *destination = input_data.data() +
                                 static_cast<size_t>(mel_index) * n_frames;

            std::copy(source, source + n_frames, destination);
        }

        input_segments.push_back(std::move(input_data));

        if (start + target_samples >= pcmf32.size())
        {
            break;
        }

        start += step_samples;
    }

    return input_segments;
}
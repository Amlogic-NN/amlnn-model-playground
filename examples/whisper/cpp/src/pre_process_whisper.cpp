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
#include "pre_post_common.h"

extern bool is_finish;

std::vector<float> do_pre_process(std::string fname_inp)
{
    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    struct whisper_context ctx;
    struct whisper_state state;

    if (!read_wav(fname_inp, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
        is_finish = true;
        return {};
    }

    if (float(pcmf32.size())/WHISPER_SAMPLE_RATE == 0) {
        is_finish = true;
        return {};
    }

    if (whisper_pcm_to_mel_with_state(&ctx, &state, pcmf32.data(), pcmf32.size(), 8) != 0) {
        printf("%s: failed to compute log mel spectrogram\n", __func__);
    }

    std::vector<float> input_data;
    for (int j = 0; j < 80; j++) {
        for (int i = 0; i < 3000; i++) {
            input_data.push_back(state.mel.data[j * state.mel.n_len + i]);
        }
    }

    return input_data;
}
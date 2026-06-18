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

#include "sense_voice.h"
#include "wav_reader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void PrintUsage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s --model <adla_model> --tokens <tokens.txt> "
                 "--lang <auto|zh|en|ja|ko|yue> --wav <input.wav> [--itn 0|1]\n",
                 prog);
}

bool ParseArgs(int argc, char **argv,
               std::string *model,
               std::string *tokens,
               std::string *lang,
               std::string *wav,
               int *use_itn) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            *model = argv[++i];
        } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            *tokens = argv[++i];
        } else if (std::strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            *lang = argv[++i];
        } else if (std::strcmp(argv[i], "--wav") == 0 && i + 1 < argc) {
            *wav = argv[++i];
        } else if (std::strcmp(argv[i], "--itn") == 0 && i + 1 < argc) {
            *use_itn = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    return !model->empty() && !tokens->empty() && !lang->empty() && !wav->empty();
}

}  // namespace

int main(int argc, char **argv) {
    std::string model;
    std::string tokens;
    std::string lang;
    std::string wav;
    int use_itn = 0;

    if (!ParseArgs(argc, argv, &model, &tokens, &lang, &wav, &use_itn)) {
        PrintUsage(argv[0]);
        return 1;
    }

    WavData wav_data;
    std::string wav_error;
    if (!ReadWavPcm16(wav, &wav_data, &wav_error)) {
        std::fprintf(stderr, "Failed to read wav: %s\n", wav_error.c_str());
        return 1;
    }

    if (wav_data.sample_rate != 16000) {
        std::fprintf(stderr,
                     "Warning: wav sample rate is %d, expected 16000. "
                     "Please provide 16 kHz audio.\n",
                     wav_data.sample_rate);
    }

    sense_voice_config_t config{};
    config.adla_model_path = model.c_str();
    config.tokens_path = tokens.c_str();
    config.use_itn = use_itn;

    sense_voice_engine_t *engine = sense_voice_create(&config);
    if (!engine) {
        std::fprintf(stderr, "Failed to create sensevoice engine\n");
        return 1;
    }

    sense_voice_result_t result{};
    if (sense_voice_recognize(
            engine, wav_data.samples.data(), wav_data.samples.size(), lang.c_str(),
            &result) != 0) {
        std::fprintf(stderr, "Recognition failed\n");
        sense_voice_destroy(engine);
        return 1;
    }

    std::printf("language: %s\n", result.language);
    std::printf("emotion:  %s\n", result.emotion);
    std::printf("event:    %s\n", result.event);
    std::printf("itn:      %s\n", result.itn);
    std::printf("text:     %s\n", result.text);
    sense_voice_free_result(&result);
    sense_voice_destroy(engine);
    return 0;
}

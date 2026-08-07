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

#include "adla_mobileclip.h"
#include "clip_tokenizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define BILLION 1000000000

struct ProfilingTimer {
    uint64_t init_start = 0;
    uint64_t init_end = 0;
    uint64_t tokenizer_start = 0;
    uint64_t tokenizer_end = 0;
    uint64_t preprocess_start = 0;
    uint64_t preprocess_end = 0;
    uint64_t image_infer_start = 0;
    uint64_t image_infer_end = 0;
    uint64_t text_infer_start = 0;
    uint64_t text_infer_end = 0;
};

static double ns_to_ms(uint64_t start, uint64_t end) {
    return static_cast<double>(end - start) / 1000000.0;
}

static void print_timing_line(const char* label, double ms) {
    printf("  %-22s: %.2f ms\n", label, ms);
}

static uint64_t get_time_count() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_nsec) + static_cast<uint64_t>(ts.tv_sec) * BILLION;
}

static std::vector<std::string> default_texts = {
    "a red handbag",
    "a blue jacket",
    "a red bus"};

static std::vector<uint8_t> load_image_rgb(const std::string& path, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    std::vector<uint8_t> rgb(data, data + width * height * 3);
    stbi_image_free(data);
    return rgb;
}

static std::vector<std::string> parse_texts(const std::string& input) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            result.push_back(item.substr(start, end - start + 1));
        }
    }
    return result;
}

static bool parse_bool_value(const char* option_name, const std::string& value, bool& out) {
    if (value == "true") {
        out = true;
        return true;
    }
    if (value == "false") {
        out = false;
        return true;
    }
    printf("[Error] %s must be true or false, got: %s\n", option_name, value.c_str());
    return false;
}

static void print_usage(const char* program) {
    printf("Usage: %s [options] <vision_model.adla> <text_model.adla> <tokenizer_dir> <image.jpg>\n",
           program);
    printf("\n");
    printf("Arguments:\n");
    printf("  vision_model:   Path to vision model (.adla)\n");
    printf("  text_model:     Path to text model (.adla)\n");
    printf("  tokenizer_dir:  Path to directory containing vocab.json and merges.txt\n");
    printf("  image_path:     Path to the image to process\n");
    printf("\n");
    printf("Options:\n");
    printf("  --neon true|false      Enable/disable NEON for ADLA soft ops (default: true)\n");
    printf("  --openmp true|false    Enable/disable OpenMP for ADLA soft ops (default: false)\n");
    printf("  --openmp-num N         OpenMP thread count when openmp is enabled (default: 2)\n");
    printf("  --profiling            Enable performance profiling output\n");
}

int main(int argc, char* argv[]) {
    bool enable_neon = true;
    bool enable_openmp = false;
    bool profiling = false;
    int openmp_num = 2;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--neon") {
            if (i + 1 >= argc) {
                printf("[Error] --neon requires true or false\n");
                print_usage(argv[0]);
                return 1;
            }
            if (!parse_bool_value("--neon", argv[++i], enable_neon)) {
                return 1;
            }
        } else if (arg == "--openmp") {
            if (i + 1 >= argc) {
                printf("[Error] --openmp requires true or false\n");
                print_usage(argv[0]);
                return 1;
            }
            if (!parse_bool_value("--openmp", argv[++i], enable_openmp)) {
                return 1;
            }
        } else if (arg == "--openmp-num") {
            if (i + 1 >= argc) {
                printf("[Error] --openmp-num requires a value\n");
                print_usage(argv[0]);
                return 1;
            }
            openmp_num = std::stoi(argv[++i]);
            if (openmp_num <= 0) {
                printf("[Error] --openmp-num must be positive\n");
                return 1;
            }
        } else if (arg == "--profiling") {
            profiling = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string image_model = positional[0];
    const std::string text_model = positional[1];
    const std::string tokenizer_dir = positional[2];
    const std::string image_path = positional[3];
    const float logit_scale = 100.0f;
    const int max_len = 77;

    ProfilingTimer timer;

    try {
        printf("[Info] Loading tokenizer from: %s\n", tokenizer_dir.c_str());
        timer.tokenizer_start = get_time_count();
        CLIPTokenizer tokenizer;
        if (!tokenizer.load_from_dir(tokenizer_dir)) {
            printf("[Error] Failed to load tokenizer.\n");
            return 1;
        }
        timer.tokenizer_end = get_time_count();

        timer.init_start = get_time_count();
        AdlaMobileCLIP::Options clip_options;
        clip_options.enable_neon = enable_neon;
        clip_options.enable_openmp = enable_openmp;
        clip_options.openmp_num = openmp_num;
        AdlaMobileCLIP clip(image_model, text_model, clip_options);
        timer.init_end = get_time_count();

        printf("[Info] Models initialized successfully.\n");

        printf("\nEnter text descriptions (comma-separated, or 'skip' for defaults):\n> ");
        std::string text_input;
        if (!std::getline(std::cin, text_input)) {
            printf("[Error] Failed to read text input.\n");
            return 1;
        }

        size_t start = text_input.find_first_not_of(" \t\r\n");
        size_t end = text_input.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            text_input = text_input.substr(start, end - start + 1);
        } else {
            text_input.clear();
        }

        std::vector<std::string> texts;
        if (text_input.empty() || text_input == "skip") {
            texts = default_texts;
        } else {
            texts = parse_texts(text_input);
        }

        if (texts.empty()) {
            printf("[Error] No texts provided.\n");
            return 1;
        }

        {
            std::ifstream img_file(image_path);
            if (!img_file.good()) {
                printf("[Error] Image not found: %s\n", image_path.c_str());
                return 1;
            }
        }

        int width = 0;
        int height = 0;
        auto rgb_data = load_image_rgb(image_path, width, height);

        timer.preprocess_start = get_time_count();
        auto image_tensor = clip.preprocess_image(rgb_data, width, height);
        timer.preprocess_end = get_time_count();

        timer.image_infer_start = get_time_count();
        auto image_feat = clip.encode_image(image_tensor);
        timer.image_infer_end = get_time_count();

        if (image_feat.empty()) {
            printf("[Error] Image model inference failed.\n");
            return 1;
        }

        std::vector<std::vector<int64_t>> token_ids;
        token_ids.reserve(texts.size());
        for (const auto& text : texts) {
            token_ids.push_back(tokenizer.encode(text, max_len));
        }

        timer.text_infer_start = get_time_count();
        auto text_feats = clip.encode_text(token_ids);
        timer.text_infer_end = get_time_count();

        if (text_feats.size() != texts.size()) {
            printf("[Error] Some text embeddings failed.\n");
            return 1;
        }

        std::vector<float> sims;
        auto probs = clip.compute_similarity(image_feat, text_feats, logit_scale, &sims);

        std::vector<std::tuple<float, float, std::string>> ranked;
        ranked.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            ranked.emplace_back(probs[i], sims[i], texts[i]);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

        printf("\n============================================================\n");
        printf("MobileCLIP-S2 Image-Text Matching Results\n");
        printf("============================================================\n");
        printf("Image: %s\n", image_path.c_str());
        printf("------------------------------------------------------------\n");

        for (size_t i = 0; i < ranked.size(); ++i) {
            printf("[%zu] prob=%.6f  sim=%.6f  text='%s'\n",
                   i + 1, std::get<0>(ranked[i]), std::get<1>(ranked[i]),
                   std::get<2>(ranked[i]).c_str());
        }
        printf("============================================================\n");

        if (profiling) {
            const double init_ms = ns_to_ms(timer.init_start, timer.init_end);
            const double tokenizer_ms = ns_to_ms(timer.tokenizer_start, timer.tokenizer_end);
            const double preprocess_ms = ns_to_ms(timer.preprocess_start, timer.preprocess_end);
            const double image_ms = ns_to_ms(timer.image_infer_start, timer.image_infer_end);
            const double text_ms = ns_to_ms(timer.text_infer_start, timer.text_infer_end);
            const double text_avg_ms = text_ms / static_cast<double>(texts.size());
            const double inference_ms = image_ms + text_ms;

            printf("\n[Timing]\n");
            print_timing_line("Model Init", init_ms);
            print_timing_line("Tokenizer Load", tokenizer_ms);
            print_timing_line("Image Preprocess", preprocess_ms);
            print_timing_line("Image Encoder", image_ms);

            char text_encoder_label[32];
            snprintf(text_encoder_label, sizeof(text_encoder_label), "Text Encoder (%zu)", texts.size());
            print_timing_line(text_encoder_label, text_ms);
            print_timing_line("Text Encoder (avg)", text_avg_ms);
            printf("  -----------------------------\n");
            print_timing_line("Inference", inference_ms);
        }
    } catch (const std::exception& e) {
        printf("[Error] %s\n", e.what());
        return 1;
    }

    return 0;
}

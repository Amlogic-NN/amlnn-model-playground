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

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>

#include "clip_process.h"
#include "clip_tokenizer.h"
#include "model_invoke.h"

#define BILLION 1000000000

struct ProfilingTimer
{
    uint64_t init_start, init_end;
    uint64_t preprocess_start, preprocess_end;
    uint64_t image_infer_start, image_infer_end;
    uint64_t text_infer_start, text_infer_end;
};

static uint64_t get_time_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
}

// Default text prompts for demo
static std::vector<std::string> default_texts = {
    "a red handbag",
    "a blue jacket",
    "a red bus"};

// Parse comma-separated texts
std::vector<std::string> parse_texts(const std::string &input)
{
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;

    while (std::getline(ss, item, ','))
    {
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
        {
            result.push_back(item.substr(start, end - start + 1));
        }
    }
    return result;
}

void print_usage(const char *prog_name)
{
    printf("Usage: %s <vision_model> <text_model> <tokenizer_dir> <image_path> [--profiling]\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  vision_model:   Path to vision model (.adla)\n");
    printf("  text_model:     Path to text model (.adla)\n");
    printf("  tokenizer_dir:  Path to directory containing vocab.json and merges.txt\n");
    printf("  image_path:     Path to the image to process\n");
    printf("  --profiling:    Enable performance profiling output (optional)\n");
    printf("\n");
    printf("Interactive mode:\n");
    printf("  - Enter comma-separated texts to compare against the image (or 'skip' for defaults)\n");
    printf("  - Enter 'exit' to quit\n");
}

int main(int argc, char **argv)
{
    ProfilingTimer timer = {};
    int ret = 0;
    bool profiling = false;

    if (argc < 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    const char *image_model_path = argv[1];
    const char *text_model_path = argv[2];
    const char *tokenizer_dir = argv[3];
    const std::string image_path = argv[4];

    for (int i = 5; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--profiling")
        {
            profiling = true;
        }
    }

    const float logit_scale = 100.0f;
    const int max_seq_len = 64;

    // Load tokenizer
    printf("[Info] Loading tokenizer from: %s\n", tokenizer_dir);
    CLIPTokenizer tokenizer;
    if (!tokenizer.load_from_dir(tokenizer_dir))
    {
        printf("[Error] Failed to load tokenizer.\n");
        return -1;
    }

    // Initialize models
    printf("[Info] Initializing image model: %s\n", image_model_path);
    timer.init_start = get_time_count();

    void *image_context = nullptr;
    if (init_network(image_model_path, image_context) != 0)
    {
        printf("[Error] Failed to initialize image model.\n");
        return -1;
    }

    printf("[Info] Initializing text model: %s\n", text_model_path);
    void *text_context = nullptr;
    if (init_network(text_model_path, text_context) != 0)
    {
        printf("[Error] Failed to initialize text model.\n");
        uninit_network(image_context);
        return -1;
    }
    timer.init_end = get_time_count();

    if (profiling)
    {
        uint64_t init_time = (timer.init_end - timer.init_start) / 1000000;
        printf("[Profiling] Model initialization: %lums\n", init_time);
    }
    printf("[Info] Models initialized successfully.\n");

    // ==================== Process Image (Done ONCE) ====================
    {
        std::ifstream img_file(image_path);
        if (!img_file.good())
        {
            printf("[Error] Image not found: %s\n", image_path.c_str());
            uninit_network(image_context);
            uninit_network(text_context);
            return -1;
        }
    }

    printf("\n[Info] Processing image: %s\n", image_path.c_str());

    timer.preprocess_start = get_time_count();
    std::vector<float> image_input = preprocess_image(image_path);
    if (image_input.empty())
    {
        printf("[Error] Failed to preprocess image.\n");
        uninit_network(image_context);
        uninit_network(text_context);
        return -1;
    }
    timer.preprocess_end = get_time_count();

    // Run image model
    timer.image_infer_start = get_time_count();
    std::vector<float> image_embedding = run_image_model(image_context, image_input);
    if (image_embedding.empty())
    {
        printf("[Error] Image model inference failed.\n");
        uninit_network(image_context);
        uninit_network(text_context);
        return -1;
    }
    timer.image_infer_end = get_time_count();

    // L2 normalize image embedding
    image_embedding = l2_normalize(image_embedding);
    printf("[Info] Image embedding size: %zu\n", image_embedding.size());
    printf("[Info] Image processed successfully. Entering interactive mode.\n");

    // ==================== Interactive Text Loop ====================
    while (true)
    {
        std::vector<std::string> texts;

        printf("\n============================================================\n");
        printf("[Info] Enter text descriptions (comma-separated, 'skip' for defaults, 'exit' to quit):\n> ");
        std::string text_input;
        if (!std::getline(std::cin, text_input))
            break;

        size_t start = text_input.find_first_not_of(" \t\r\n");
        size_t end = text_input.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos)
        {
            text_input = text_input.substr(start, end - start + 1);
        }
        else
        {
            text_input.clear();
        }

        if (text_input == "exit")
        {
            printf("[Info] Exiting...\n");
            break;
        }

        if (text_input.empty() || text_input == "skip")
        {
            texts = default_texts;
            printf("[Info] Using default texts\n");
        }
        else
        {
            texts = parse_texts(text_input);
        }

        if (texts.empty())
        {
            printf("[Warning] No texts provided.\n");
            continue;
        }

        // ==================== Process Texts ====================
        printf("[Info] Processing %zu text(s)...\n", texts.size());

        std::vector<std::vector<float>> text_embeddings;
        std::vector<uint64_t> text_infer_times;
        timer.text_infer_start = get_time_count();

        for (size_t i = 0; i < texts.size(); ++i)
        {
            // Tokenize text
            std::vector<int64_t> token_ids = tokenizer.encode(texts[i], max_seq_len);

            uint64_t t_start = get_time_count();
            std::vector<float> text_emb = run_text_model(text_context, token_ids);
            uint64_t t_end = get_time_count();
            text_infer_times.push_back((t_end - t_start) / 1000000);

            if (text_emb.empty())
            {
                printf("[Error] Text model inference failed for: %s\n", texts[i].c_str());
                continue;
            }

            // L2 normalize
            text_emb = l2_normalize(text_emb);
            text_embeddings.push_back(text_emb);
        }

        timer.text_infer_end = get_time_count();

        if (text_embeddings.size() != texts.size())
        {
            printf("[Error] Some text embeddings failed.\n");
            continue;
        }

        // ==================== Compute Similarity ====================
        std::vector<float> similarities(texts.size());
        std::vector<float> logits(texts.size());

        for (size_t i = 0; i < texts.size(); ++i)
        {
            similarities[i] = compute_similarity(image_embedding, text_embeddings[i], 1.0f);
            logits[i] = similarities[i] * logit_scale;
        }

        // Compute probabilities
        std::vector<float> probs = softmax(logits);

        // Sort by probability (descending)
        std::vector<size_t> indices(texts.size());
        for (size_t i = 0; i < texts.size(); ++i)
            indices[i] = i;
        std::sort(indices.begin(), indices.end(),
                  [&probs](size_t a, size_t b)
                  { return probs[a] > probs[b]; });

        // ==================== Print Results ====================
        printf("\n============================================================\n");
        printf("CLIP Image-Text Matching Results\n");
        printf("============================================================\n");
        printf("Image: %s\n", image_path.c_str());
        printf("logit_scale: %.6f\n", logit_scale);
        printf("------------------------------------------------------------\n");

        for (size_t rank = 0; rank < indices.size(); ++rank)
        {
            size_t i = indices[rank];
            printf("[%zu] prob=%.6f  sim=%.6f  text='%s'\n",
                   rank + 1, probs[i], similarities[i], texts[i].c_str());
        }
        printf("============================================================\n");

        if (profiling)
        {
            uint64_t preprocess_time = (timer.preprocess_end - timer.preprocess_start) / 1000000;
            uint64_t image_time = (timer.image_infer_end - timer.image_infer_start) / 1000000;
            uint64_t text_total_time = (timer.text_infer_end - timer.text_infer_start) / 1000000;
            printf("\n[Profiling]\n");
            printf("  Image preprocess (Done Once):  %lums\n", preprocess_time);
            printf("  Image inference  (Done Once):  %lums\n", image_time);
            for (size_t i = 0; i < texts.size() && i < text_infer_times.size(); ++i)
            {
                printf("  Text inference[%zu]: %lums  '%s'\n", i, text_infer_times[i], texts[i].c_str());
            }
            printf("  Text total for this prompt:  %lums (%zu texts)\n", text_total_time, texts.size());
        }
    }

    // Cleanup
    if (uninit_network(image_context) != 0)
        printf("[Error] Failed to destroy image model.\n");
    if (uninit_network(text_context) != 0)
        printf("[Error] Failed to destroy text model.\n");

    printf("[Info] Done.\n");
    return 0;
}
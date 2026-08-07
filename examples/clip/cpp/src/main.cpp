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
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <string>
#include <cctype>
#include <vector>
#include "postprocess.h"
#include "clip_tokenizer.h"
#include "model_invoke.h"
#include "model_loader.h"
#include "nnsdk2.h"

const float LOGIT_SCALE = 100.0f;
const int MAX_TEXT_LENGTH = 64;
namespace fs = std::filesystem;

static const std::vector<std::string> DEFAULT_TEXTS = {
    "a red handbag",
    "a blue jacket",
    "a red bus"};

static void print_usage(const char *program)
{
    std::cout << "Usage: " << program << " <image_model.adla> <text_model.adla> <tokenizer_dir> <image_dir> [texts ...]" << std::endl;
    std::cout << "Multi-word texts must be quoted." << std::endl;
    std::cout << "Example: " << program << " clip_image.adla clip_text.adla tokenizer images \"a red handbag\" \"a blue jacket\" \"a red bus\"" << std::endl;
    std::cout << "Default texts are used when no texts are provided." << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        print_usage(argv[0]);
        return 0;
    }

    std::string image_model_path = argv[1];
    std::string text_model_path = argv[2];
    std::string tokenizer_dir = argv[3];
    std::string image_dir = argv[4];
    std::vector<std::string> texts;

    for (int i = 5; i < argc; ++i)
        texts.push_back(argv[i]);

    if (texts.empty())
        texts = DEFAULT_TEXTS;

    if (!fs::exists(image_dir) || !fs::is_directory(image_dir))
    {
        std::cerr << "Invalid image directory: " << image_dir << std::endl;
        return -1;
    }

    CLIPTokenizer tokenizer;
    if (!tokenizer.load_from_dir(tokenizer_dir))
    {
        std::cerr << "Failed to load tokenizer from: " << tokenizer_dir << std::endl;
        return -1;
    }

    void *image_context = nullptr;
    if (init_network(image_model_path, image_context) != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize image model" << std::endl;
        return -1;
    }

    void *text_context = nullptr;
    if (init_network(text_model_path, text_context) != AMLNN_SUCCESS)
    {
        std::cerr << "Failed to initialize text model" << std::endl;
        uninit_network(image_context);
        return -1;
    }

    amlnn_tensor_attr image_input_attr = query_input_attr(image_context, 0);
    amlnn_tensor_attr image_output_attr = query_output_attr(image_context, 0);
    amlnn_tensor_attr text_input_attr = query_input_attr(text_context, 0);
    amlnn_tensor_attr text_output_attr = query_output_attr(text_context, 0);

    std::cout << "image input elements: " << image_input_attr.n_elems << ", output elements: " << image_output_attr.n_elems << std::endl;
    std::cout << "Text input elements: " << text_input_attr.n_elems << ", output elements: " << text_output_attr.n_elems << std::endl;
    std::cout << "Text prompts (" << texts.size() << "):";
    for (const std::string &text : texts)
        std::cout << " \"" << text << "\"";
    std::cout << std::endl;

    std::vector<std::vector<float>> text_embeddings;
    for (const std::string &text : texts)
    {
        std::vector<int64_t> token_ids = tokenizer.encode(text, MAX_TEXT_LENGTH);
        std::vector<float> embedding = run_text_model(text_context, token_ids);

        if (embedding.empty())
        {
            std::cerr << "Text model inference failed for: " << text << std::endl;
            uninit_network(image_context);
            uninit_network(text_context);
            return -1;
        }

        text_embeddings.push_back(l2_normalize(embedding));
    }

    std::vector<fs::path> image_paths;
    for (const auto &it : fs::directory_iterator(image_dir))
    {
        if (it.is_regular_file())
            image_paths.push_back(it.path());
    }

    std::sort(image_paths.begin(), image_paths.end());

    int processed_images = 0;

    for (const fs::path &path : image_paths)
    {
        if (!fs::is_regular_file(path))
            continue;

        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c)
                       { return std::tolower(c); });

        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png" && extension != ".bmp")
            continue;

        std::string image_path = path.string();

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: " << path.filename().string() << std::endl;
        std::cout << "============================================================" << std::endl;

        std::vector<float> image_input = preprocess_image(image_path);
        if (image_input.empty())
        {
            std::cerr << "Failed to preprocess image: " << image_path << std::endl;
            continue;
        }

        std::vector<float> image_embedding = run_image_model(image_context, image_input);
        if (image_embedding.empty())
        {
            std::cerr << "image model inference failed: " << image_path << std::endl;
            continue;
        }

        image_embedding = l2_normalize(image_embedding);

        std::vector<float> similarities(texts.size());
        std::vector<float> logits(texts.size());

        for (size_t i = 0; i < texts.size(); ++i)
        {
            similarities[i] = compute_similarity(image_embedding, text_embeddings[i], 1.0f);
            logits[i] = similarities[i] * LOGIT_SCALE;
        }

        std::vector<float> probabilities = softmax(logits);
        std::vector<size_t> indices(texts.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&probabilities](size_t a, size_t b)
                  { return probabilities[a] > probabilities[b]; });

        std::cout << "CLIP Image-Text Matching Results" << std::endl;
        std::cout << std::fixed << std::setprecision(6);

        for (size_t rank = 0; rank < indices.size(); ++rank)
        {
            size_t i = indices[rank];
            std::cout << "[" << rank + 1 << "] prob=" << probabilities[i] << " sim=" << similarities[i] << " text=\"" << texts[i] << "\"" << std::endl;
        }

        ++processed_images;
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "Processed " << processed_images << " image(s)" << std::endl;

    uninit_network(image_context);
    uninit_network(text_context);
    return 0;
}
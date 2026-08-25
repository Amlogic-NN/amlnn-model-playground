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
#include <vector>
#include <cmath>
#include <fstream>

#include "postprocess.h"
#include "siglip_tokenizer.h"
#include "model_invoke.h"
#include "model_loader.h"
#include "nnsdk2.h"

namespace fs = std::filesystem;

// SigLIP specific constants
const float LOGIT_SCALE = 4.724453449249268f;
const float LOGIT_BIAS = -16.771724700927734f;
const int TOP_K = 5;

static const std::vector<std::string> DEFAULT_TEXTS = {
    "a red handbag",
    "a blue jacket",
    "a red bus"
};

struct Result {
    std::string label;
    float probability;
    float logit;
};

static void print_usage(const char *program)
{
    std::cout << "Usage: " << program << " <image_model.adla> <text_model.adla> <data_bin> <image_dir> [texts ...]\n";
    std::cout << "Multi-word texts must be quoted.\n";
    std::cout << "Example: " << program << " siglip_image.adla siglip_text.adla ./data_bin ../input \"a red handbag\" \"a blue jacket\" \"a red bus\"\n";
}

std::string format_prompt(const std::string& tmpl, const std::string& label) {
    std::string result = tmpl;
    size_t pos = result.find("{}");
    if (pos != std::string::npos) {
        result.replace(pos, 2, label);
    }
    return result;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        print_usage(argv[0]);
        return 0;
    }

    std::string image_model = argv[1];
    std::string text_model = argv[2];
    std::string tokenizer_dir = argv[3];
    std::string image_dir = argv[4];
    std::vector<std::string> prompts;

    for (int i = 5; i < argc; ++i) {
        prompts.push_back(argv[i]);
    }

    if (prompts.empty()) prompts = DEFAULT_TEXTS;

    if (!fs::exists(image_dir) || !fs::is_directory(image_dir)) {
        std::cerr << "Invalid image directory: " << image_dir << std::endl;
        return -1;
    }

    std::string prompt_template = "This is a photo of {}.";

    // --- USE NEW TOKENIZER ---
    SigLIPTokenizer tokenizer;
    if (!tokenizer.load_from_dir(tokenizer_dir)) {
        std::cerr << "Failed to load tokenizer from: " << tokenizer_dir << std::endl;
        return -1;
    }

    void *image_context = nullptr;
    if (init_network(image_model, image_context) != AMLNN_SUCCESS) return -1;

    void *text_context = nullptr;
    if (init_network(text_model, text_context) != AMLNN_SUCCESS) {
        uninit_network(image_context);
        return -1;
    }

    amlnn_tensor_attr image_input_attr = query_input_attr(image_context, 0);
    amlnn_tensor_attr text_input_attr = query_input_attr(text_context, 0);

    int image_dim = std::sqrt(image_input_attr.n_elems / 3);
    int text_length = text_input_attr.n_elems;

    std::cout << "Image input size (HxW): " << image_dim << "x" << image_dim << "\n";
    std::cout << "Text length: " << text_length << "\n\n";

    // 1. Process Text Embeddings
    std::vector<std::vector<float>> text_embeddings;
    for (const std::string &label : prompts)
    {
        std::string prompt = format_prompt(prompt_template, label);
        std::vector<int64_t> token_ids = tokenizer.encode(prompt, text_length);
        std::vector<float> embedding = run_text_model(text_context, token_ids);

        if (embedding.empty()) {
            std::cerr << "Text model inference failed for: " << prompt << std::endl;
            return -1;
        }
        text_embeddings.push_back(embedding);
    }

    // 2. Discover Images
    std::vector<fs::path> image_files;
    for (const auto &it : fs::directory_iterator(image_dir))
    {
        if (!it.is_regular_file()) continue;
        std::string ext = it.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            image_files.push_back(it.path());
    }

    std::sort(image_files.begin(), image_files.end());

    fs::path image_path(image_model);
    fs::path result_dir = image_path.stem().string() + "_result";
    fs::create_directories(result_dir);

    // 3. Process each Image
    int processed = 0;
    float exp_logit_scale = std::exp(LOGIT_SCALE);

    for (const fs::path &path : image_files)
    {
        std::string image_path = path.string();

        std::cout << std::string(60, '=') << "\n";
        std::cout << "Processing image " << (processed + 1) << "/" << image_files.size() << ": " << path.filename().string() << "\n";
        std::cout << std::string(60, '=') << "\n";

        std::vector<float> image_input = preprocess_image(image_path, image_dim, image_dim);
        if (image_input.empty()) continue;

        std::vector<float> image_embedding = run_image_model(image_context, image_input);
        if (image_embedding.empty()) continue;

        std::vector<Result> results;
        for (size_t i = 0; i < prompts.size(); ++i)
        {
            float dot = compute_similarity(image_embedding, text_embeddings[i]);
            float logit = dot * exp_logit_scale + LOGIT_BIAS;
            float prob = sigmoid(logit);

            results.push_back({prompts[i], prob, logit});
        }

        std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
            return a.probability > b.probability;
        });

        int k = std::min((int)TOP_K, (int)results.size());
        fs::path save_path = result_dir / (path.stem().string() + "_result.txt");
        std::ofstream ofs(save_path);

        for (int rank = 0; rank < k; ++rank)
        {
            std::cout << "    [" << rank + 1 << "] " << results[rank].label << ": "
                      << "probability=" << std::fixed << std::setprecision(6) << results[rank].probability
                      << ", logit=" << results[rank].logit << "\n";

            ofs << rank + 1 << ". " << results[rank].label << ": "
                << "probability=" << std::fixed << std::setprecision(6) << results[rank].probability
                << ", logit=" << results[rank].logit << "\n";
        }

        std::cout << "    Result saved to: " << save_path.string() << "\n\n";
        ++processed;
    }

    std::cout << std::string(60, '=') << "\n";
    uninit_network(image_context);
    uninit_network(text_context);
    return 0;
}
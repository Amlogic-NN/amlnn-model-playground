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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

// Implemented in pre_postprocess.cpp
std::vector<float> preprocess_image_impl(const std::vector<uint8_t>& rgb_data, int width, int height);

namespace {

std::vector<uint8_t> chw_float_to_model_input(const std::vector<float>& chw,
                                              const ModelInputTensorDesc& desc) {
    if (desc.dim_count != 4) return {};

    const int C = 3;
    const bool is_nchw = desc.tensor_format == AMLNN_TENSOR_NCHW;
    const int H = static_cast<int>(is_nchw ? desc.dims[2] : desc.dims[1]);
    const int W = static_cast<int>(is_nchw ? desc.dims[3] : desc.dims[2]);
    if (H <= 0 || W <= 0 || chw.size() != static_cast<size_t>(C) * H * W) return {};

    const size_t elem_count = static_cast<size_t>(H) * W * C;
    const bool is_fp32 = desc.tensor_type == AMLNN_TENSOR_FLOAT32;
    const bool is_int16 = desc.tensor_type == AMLNN_TENSOR_INT16;
    const bool is_int8 = desc.tensor_type == AMLNN_TENSOR_INT8;
    const bool is_uint8 = desc.tensor_type == AMLNN_TENSOR_UINT8;
    if (!is_fp32 && !is_int16 && !is_int8 && !is_uint8) return {};

    const size_t bytes_per_elem = is_fp32 ? sizeof(float)
                                          : (is_int16 ? sizeof(int16_t) : sizeof(uint8_t));
    std::vector<uint8_t> raw(elem_count * bytes_per_elem, 0);

    auto write_value = [&](size_t index, float val) {
        if (is_fp32) {
            reinterpret_cast<float*>(raw.data())[index] = val;
            return;
        }

        const float safe_scale = std::abs(desc.scale) > 1e-12f ? desc.scale : 1.0f;
        int32_t q = static_cast<int32_t>(std::round(val / safe_scale) + desc.zero_point);
        if (is_int16) {
            q = std::max(-32768, std::min(32767, q));
            reinterpret_cast<int16_t*>(raw.data())[index] = static_cast<int16_t>(q);
        } else if (is_int8) {
            q = std::max(-128, std::min(127, q));
            reinterpret_cast<int8_t*>(raw.data())[index] = static_cast<int8_t>(q);
        } else {
            q = std::max(0, std::min(255, q));
            raw[index] = static_cast<uint8_t>(q);
        }
    };

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                const float val = chw[c * H * W + h * W + w];
                const size_t dst_index = is_nchw
                    ? static_cast<size_t>(c) * H * W + h * W + w
                    : (static_cast<size_t>(h) * W + w) * C + c;
                write_value(dst_index, val);
            }
        }
    }
    return raw;
}

AdlaInitOptions to_init_options(const AdlaMobileCLIP::Options& options) {
    AdlaInitOptions init_options;
    init_options.enable_neon = options.enable_neon;
    init_options.enable_openmp = options.enable_openmp;
    init_options.openmp_num = options.openmp_num;
    return init_options;
}

}  // namespace

AdlaMobileCLIP::AdlaMobileCLIP(const std::string& image_adla_path,
                               const std::string& text_adla_path)
    : AdlaMobileCLIP(image_adla_path, text_adla_path, Options{}) {
}

AdlaMobileCLIP::AdlaMobileCLIP(const std::string& image_adla_path,
                               const std::string& text_adla_path,
                               const Options& options) {
    const AdlaInitOptions init_options = to_init_options(options);

    printf("[Info] Initializing image model: %s\n", image_adla_path.c_str());
    image_context_ = init_network_file(image_adla_path.c_str(), init_options);
    if (!image_context_) {
        throw std::runtime_error("Failed to init image ADLA model: " + image_adla_path);
    }

    printf("[Info] Initializing text model: %s\n", text_adla_path.c_str());
    text_context_ = init_network_file(text_adla_path.c_str(), init_options);
    if (!text_context_) {
        destroy_network(image_context_);
        image_context_ = nullptr;
        throw std::runtime_error("Failed to init text ADLA model: " + text_adla_path);
    }
}

AdlaMobileCLIP::~AdlaMobileCLIP() {
    if (image_context_) destroy_network(image_context_);
    if (text_context_) destroy_network(text_context_);
}

std::vector<float> AdlaMobileCLIP::preprocess_image(const std::vector<uint8_t>& rgb_data,
                                                    int width,
                                                    int height) {
    return preprocess_image_impl(rgb_data, width, height);
}

std::vector<float> AdlaMobileCLIP::encode_image(const std::vector<float>& image_data) {
    if (!image_context_ || image_data.empty()) return {};

    ModelInputTensorDesc input_desc;
    if (!get_input_tensor_desc(image_context_, 0, &input_desc)) {
        std::cerr << "[Error] Failed to query image input tensor info." << std::endl;
        return {};
    }

    std::vector<uint8_t> model_input = chw_float_to_model_input(image_data, input_desc);
    if (model_input.empty()) {
        std::cerr << "[Error] Failed to convert image tensor for ADLA input." << std::endl;
        return {};
    }

    return run_image_model(image_context_, model_input);
}

std::vector<std::vector<float>> AdlaMobileCLIP::encode_text(
    const std::vector<std::vector<int64_t>>& token_ids) {
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(token_ids.size());

    for (const auto& tokens : token_ids) {
        auto feat = run_text_model(text_context_, tokens);
        if (!feat.empty()) {
            embeddings.push_back(std::move(feat));
        }
    }
    return embeddings;
}

std::vector<float> AdlaMobileCLIP::compute_similarity(
    const std::vector<float>& image_feat,
    const std::vector<std::vector<float>>& text_feats,
    float logit_scale,
    std::vector<float>* out_sims) {
    auto normalize = [](std::vector<float>& v) {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm + 1e-12f);
        for (float& x : v) x /= norm;
    };

    std::vector<float> img = image_feat;
    normalize(img);

    std::vector<float> logits;
    if (out_sims) out_sims->clear();

    for (auto txt : text_feats) {
        normalize(txt);
        float sim = 0.0f;
        for (size_t i = 0; i < img.size(); ++i) {
            sim += img[i] * txt[i];
        }
        if (out_sims) out_sims->push_back(sim);
        logits.push_back(sim * logit_scale);
    }

    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (float& v : logits) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (float& v : logits) v /= sum;

    return logits;
}

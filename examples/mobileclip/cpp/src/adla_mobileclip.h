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

#ifndef ADLA_MOBILECLIP_H
#define ADLA_MOBILECLIP_H

#include <string>
#include <vector>
#include <cstdint>

#include "model_invoke.h"

// ADLA-backed MobileCLIP inference (NPU)
class AdlaMobileCLIP {
public:
    struct Options {
        bool enable_neon = true;
        bool enable_openmp = false;
        int openmp_num = 2;
    };

    AdlaMobileCLIP(const std::string& image_adla_path,
                   const std::string& text_adla_path);
    AdlaMobileCLIP(const std::string& image_adla_path,
                   const std::string& text_adla_path,
                   const Options& options);
    ~AdlaMobileCLIP();

    std::vector<float> preprocess_image(const std::vector<uint8_t>& rgb_data, int width, int height);
    std::vector<float> encode_image(const std::vector<float>& image_data);
    std::vector<std::vector<float>> encode_text(const std::vector<std::vector<int64_t>>& token_ids);
    std::vector<float> compute_similarity(const std::vector<float>& image_feat,
                                          const std::vector<std::vector<float>>& text_feats,
                                          float logit_scale = 100.0f,
                                          std::vector<float>* out_sims = nullptr);

    std::string backend_summary() const { return "ADLA"; }

private:
    void* image_context_ = nullptr;
    void* text_context_ = nullptr;
};

#endif // ADLA_MOBILECLIP_H
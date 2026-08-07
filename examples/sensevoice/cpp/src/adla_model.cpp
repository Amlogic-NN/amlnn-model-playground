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

#include "adla_model.h"
#include "constants.h"

#include "nnsdk2.h"

#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kInputTextNorm = 0;
constexpr uint32_t kInputLanguage = 1;
constexpr uint32_t kInputFeatures = 2;
constexpr uint32_t kOutputLogits = 0;
constexpr uint32_t kExpectedInputs = 3;
constexpr uint32_t kExpectedOutputs = 1;

bool ValidateModelIo(void *ctx) {
    amlnn_input_output_num io{};
    if (amlnn_query(ctx, AMLNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != AMLNN_SUCCESS) {
        return false;
    }
    return io.n_input == kExpectedInputs && io.n_output == kExpectedOutputs;
}

bool SetFeatureShape(void *ctx, int32_t num_frames, int32_t feature_dim) {
    amlnn_tensor_attr feat{};
    feat.index = kInputFeatures;
    feat.n_dims = 3;
    feat.dims[0] = 1;
    feat.dims[1] = static_cast<uint32_t>(num_frames);
    feat.dims[2] = static_cast<uint32_t>(feature_dim);
    return amlnn_set_input_shapes(ctx, 1, &feat) == AMLNN_SUCCESS;
}

void EnableSoftopAcceleration(void *ctx) {
    amlnn_softop_opt_request req[] = {{
        AMLNN_Unknown,
        AMLNN_SOFTOP_ACC_OPENMP,
    }};
    amlnn_set_softop_opt(ctx, req, 1);
}

}  // namespace

AdlaModel::AdlaModel(const std::string &adla_model_path) {
    amlnn_init_config cfg{};
    cfg.backend_type = AMLNN_BACKEND_ADLA_NPU;

    void *ctx = nullptr;
    if (amlnn_init(&ctx, const_cast<char *>(adla_model_path.c_str()), 0, &cfg) !=
            AMLNN_SUCCESS ||
        ctx == nullptr) {
        return;
    }

    if (!ValidateModelIo(ctx)) {
        amlnn_destroy(ctx);
        return;
    }

    EnableSoftopAcceleration(ctx);
    context_ = ctx;
    ok_ = true;
}

AdlaModel::~AdlaModel() {
    if (context_) {
        amlnn_destroy(context_);
        context_ = nullptr;
    }
}

AdlaForwardOutput AdlaModel::Forward(const float *features,
                                     int32_t num_frames,
                                     int32_t feature_dim,
                                     int32_t language_id,
                                     int32_t text_norm_id) {
    AdlaForwardOutput output;
    if (!ok_ || !context_ || !features || num_frames <= 0 || feature_dim <= 0) {
        return output;
    }

    if (!SetFeatureShape(context_, num_frames, feature_dim)) {
        return output;
    }

    int32_t text_norm = text_norm_id;
    int32_t language = language_id;

    amlnn_input inputs[kExpectedInputs]{};
    inputs[0].index = kInputTextNorm;
    inputs[0].buf = &text_norm;
    inputs[0].size = sizeof(int32_t);

    inputs[1].index = kInputLanguage;
    inputs[1].buf = &language;
    inputs[1].size = sizeof(int32_t);

    inputs[2].index = kInputFeatures;
    inputs[2].buf = const_cast<float *>(features);
    inputs[2].size = static_cast<uint32_t>(
        static_cast<size_t>(num_frames) * static_cast<size_t>(feature_dim) * sizeof(float));

    if (amlnn_inputs_set(context_, kExpectedInputs, inputs) != AMLNN_SUCCESS) {
        return output;
    }
    if (amlnn_run(context_, nullptr) != AMLNN_SUCCESS) {
        return output;
    }

    amlnn_output out{};
    out.index = kOutputLogits;
    out.is_float = 1;
    if (amlnn_outputs_get(context_, kExpectedOutputs, &out) != AMLNN_SUCCESS) {
        return output;
    }
    if (!out.buf || out.size == 0 || out.size % sizeof(float) != 0) {
        return output;
    }

    const int32_t elem_count = static_cast<int32_t>(out.size / sizeof(float));
    output.vocab_size = kVocabSize;
    output.num_frames = elem_count / output.vocab_size;
    output.logits.resize(static_cast<size_t>(elem_count));
    std::memcpy(output.logits.data(), out.buf, out.size);
    return output;
}

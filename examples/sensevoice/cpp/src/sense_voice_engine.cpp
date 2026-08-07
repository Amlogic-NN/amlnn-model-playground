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

#include "adla_model.h"
#include "constants.h"
#include "ctc_decoder.h"
#include "feature_extractor.h"
#include "symbol_table.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct sense_voice_engine {
    sense_voice_config_t config;
    std::string adla_model_path;
    std::string tokens_path;
    std::unique_ptr<AdlaModel> model;
    std::unique_ptr<SymbolTable> sym_table;
    CtcGreedyDecoder decoder{kBlankId};
};

namespace {

char *DupString(const std::string &text) {
    char *out = static_cast<char *>(std::malloc(text.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

struct ParsedResult {
    std::string language;
    std::string emotion;
    std::string event;
    std::string itn;
    std::string text;
};

ParsedResult SplitTokens(const std::vector<int32_t> &token_ids,
                         const SymbolTable &sym_table) {
    ParsedResult out;
    if (token_ids.size() >= 1) {
        out.language = sym_table.Lookup(token_ids[0]);
    }
    if (token_ids.size() >= 2) {
        out.emotion = sym_table.Lookup(token_ids[1]);
    }
    if (token_ids.size() >= 3) {
        out.event = sym_table.Lookup(token_ids[2]);
    }
    if (token_ids.size() >= 4) {
        out.itn = sym_table.Lookup(token_ids[3]);
    }

    out.text.reserve(token_ids.size() * 4);
    for (size_t i = kMetaTokenCount; i < token_ids.size(); ++i) {
        out.text += sym_table.Lookup(token_ids[i]);
    }
    return out;
}

bool FillResult(const ParsedResult &parsed, sense_voice_result_t *result) {
    result->language = DupString(parsed.language);
    result->emotion = DupString(parsed.emotion);
    result->event = DupString(parsed.event);
    result->itn = DupString(parsed.itn);
    result->text = DupString(parsed.text);

    if (!result->language || !result->emotion || !result->event || !result->itn ||
        !result->text) {
        sense_voice_free_result(result);
        return false;
    }
    return true;
}

}  // namespace

sense_voice_engine_t *sense_voice_create(const sense_voice_config_t *config) {
    if (!config || !config->adla_model_path || !config->tokens_path) {
        return nullptr;
    }

    auto engine = std::make_unique<sense_voice_engine_t>();
    engine->config = *config;
    engine->adla_model_path = config->adla_model_path;
    engine->tokens_path = config->tokens_path;

    engine->model = std::make_unique<AdlaModel>(engine->adla_model_path);
    if (!engine->model->Ok()) {
        return nullptr;
    }

    engine->sym_table = std::make_unique<SymbolTable>(engine->tokens_path);
    if (!engine->sym_table->Ok()) {
        return nullptr;
    }

    return engine.release();
}

void sense_voice_destroy(sense_voice_engine_t *engine) {
    delete engine;
}

int sense_voice_recognize(sense_voice_engine_t *engine,
                          const int16_t *pcm,
                          size_t num_samples,
                          const char *language,
                          sense_voice_result_t *result) {
    if (!engine || !engine->model || !engine->sym_table || !pcm || num_samples == 0 ||
        !result) {
        return -1;
    }

    std::memset(result, 0, sizeof(*result));

    std::vector<float> float_pcm(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float_pcm[i] = static_cast<float>(pcm[i]) / 32768.f;
    }

    std::vector<float> features = ExtractFeatures(
        float_pcm.data(), static_cast<int32_t>(float_pcm.size()));

    const int32_t language_id = LanguageToId(language);
    const int32_t text_norm_id =
        engine->config.use_itn ? kWithItnId : kWithoutItnId;

    AdlaForwardOutput forward = engine->model->Forward(
        features.data(), kFixedFrames, kLfrOutDim,
        language_id, text_norm_id);
    if (forward.logits.empty() || forward.num_frames <= 0 || forward.vocab_size <= 0) {
        return -1;
    }

    std::vector<int32_t> token_ids = engine->decoder.Decode(
        forward.logits.data(), forward.num_frames, forward.vocab_size);

    const ParsedResult parsed = SplitTokens(token_ids, *engine->sym_table);
    if (!FillResult(parsed, result)) {
        return -1;
    }
    return 0;
}

void sense_voice_free_result(sense_voice_result_t *result) {
    if (!result) {
        return;
    }
    std::free(result->language);
    std::free(result->emotion);
    std::free(result->event);
    std::free(result->itn);
    std::free(result->text);
    std::memset(result, 0, sizeof(*result));
}

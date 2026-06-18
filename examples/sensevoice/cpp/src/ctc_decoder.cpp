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

#include "ctc_decoder.h"

#include <algorithm>

std::vector<int32_t> CtcGreedyDecoder::Decode(const float *logits,
                                                int32_t num_frames,
                                                int32_t vocab_size) const {
    std::vector<int32_t> tokens;
    if (!logits || num_frames <= 0 || vocab_size <= 0) {
        return tokens;
    }

    int32_t prev_id = -1;
    tokens.reserve(num_frames);

    for (int32_t t = 0; t < num_frames; ++t) {
        const float *row = logits + static_cast<size_t>(t) * vocab_size;
        int32_t best_id = static_cast<int32_t>(
            std::distance(row, std::max_element(row, row + vocab_size)));

        if (best_id != blank_id_ && best_id != prev_id) {
            tokens.push_back(best_id);
        }
        prev_id = best_id;
    }

    return tokens;
}

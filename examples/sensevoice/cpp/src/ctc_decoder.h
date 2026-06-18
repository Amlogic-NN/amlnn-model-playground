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

#ifndef SENSEVOICE_INTERNAL_CTC_DECODER_H_
#define SENSEVOICE_INTERNAL_CTC_DECODER_H_

#include <cstdint>
#include <vector>

class CtcGreedyDecoder {
public:
    explicit CtcGreedyDecoder(int32_t blank_id) : blank_id_(blank_id) {}

    std::vector<int32_t> Decode(const float *logits,
                                int32_t num_frames,
                                int32_t vocab_size) const;

private:
    int32_t blank_id_;
};

#endif  // SENSEVOICE_INTERNAL_CTC_DECODER_H_

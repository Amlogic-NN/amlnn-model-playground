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

#ifndef SENSEVOICE_INTERNAL_CONSTANTS_H_
#define SENSEVOICE_INTERNAL_CONSTANTS_H_

#include <cstdint>

constexpr int32_t kSampleRate = 16000;
constexpr int32_t kFeatureDim = 80;
constexpr int32_t kLfrWindowSize = 7;
constexpr int32_t kLfrWindowShift = 6;
constexpr int32_t kLfrOutDim = 560;
constexpr int32_t kFixedFrames = 100;
constexpr int32_t kVocabSize = 25055;
constexpr int32_t kBlankId = 0;
constexpr int32_t kMetaTokenCount = 4;
constexpr int32_t kWithItnId = 14;
constexpr int32_t kWithoutItnId = 15;

inline int32_t LanguageToId(const char *language) {
    if (!language) {
        return 0;
    }
    if (language[0] == 'a') {
        return 0;  // auto
    }
    if (language[0] == 'z' && language[1] == 'h') {
        return 3;
    }
    if (language[0] == 'e' && language[1] == 'n') {
        return 4;
    }
    if (language[0] == 'y') {
        return 7;  // yue
    }
    if (language[0] == 'j' && language[1] == 'a') {
        return 11;
    }
    if (language[0] == 'k' && language[1] == 'o') {
        return 12;
    }
    return 0;
}

#endif  // SENSEVOICE_INTERNAL_CONSTANTS_H_

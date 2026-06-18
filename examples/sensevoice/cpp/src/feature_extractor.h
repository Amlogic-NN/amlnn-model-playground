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

#ifndef SENSEVOICE_INTERNAL_FEATURE_EXTRACTOR_H_
#define SENSEVOICE_INTERNAL_FEATURE_EXTRACTOR_H_

#include <cstdint>
#include <vector>

// Returns row-major features with shape (kFixedFrames, kLfrOutDim).
std::vector<float> ExtractFeatures(const float *samples, int32_t num_samples);

#endif  // SENSEVOICE_INTERNAL_FEATURE_EXTRACTOR_H_

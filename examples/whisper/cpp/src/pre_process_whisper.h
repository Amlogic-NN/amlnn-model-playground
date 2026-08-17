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

#ifndef PRE_PROCESS_WHISPER_H
#define PRE_PROCESS_WHISPER_H

#include <string>
#include <vector>

std::vector<std::vector<float>> do_pre_process(
    const std::string &fname_inp,
    int n_mel,
    int n_frames,
    int overlap_seconds
);

#endif // PRE_PROCESS_WHISPER_H
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

#ifndef POST_PROCESS_WHISPER_H
#define POST_PROCESS_WHISPER_H

#include <cstdint>
#include <string>
#include <vector>

#include "whisper.h"

whisper_vocab read_token_info(const std::string &token_path);
std::string decode_tokens(const std::vector<int64_t> &token_ids, const whisper_vocab &vocab);
std::string merge_transcriptions(const std::vector<std::string> &transcriptions);

#endif // POST_PROCESS_WHISPER_H
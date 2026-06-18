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

#ifndef SENSE_VOICE_H_
#define SENSE_VOICE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sense_voice_engine sense_voice_engine_t;

typedef struct {
    const char *adla_model_path;
    const char *tokens_path;
    int use_itn;
} sense_voice_config_t;

/** Recognition output with meta tags separated from transcript text. */
typedef struct {
    char *language;  /**< LID, e.g. "<|en|>" */
    char *emotion;   /**< SER, e.g. "<|NEUTRAL|>" */
    char *event;     /**< AED, e.g. "<|Speech|>" */
    char *itn;       /**< ITN flag, e.g. "<|withitn|>" or "<|woitn|>" */
    char *text;      /**< ASR transcript without meta tags */
} sense_voice_result_t;

sense_voice_engine_t *sense_voice_create(const sense_voice_config_t *config);
void sense_voice_destroy(sense_voice_engine_t *engine);

/**
 * Recognize 16 kHz mono PCM (int16).
 *
 * @param language One of: "auto", "zh", "en", "ja", "ko", "yue"
 * @param result   Output structure. Caller must free with sense_voice_free_result().
 * @return 0 on success, -1 on failure.
 */
int sense_voice_recognize(sense_voice_engine_t *engine,
                          const int16_t *pcm,
                          size_t num_samples,
                          const char *language,
                          sense_voice_result_t *result);

void sense_voice_free_result(sense_voice_result_t *result);

#ifdef __cplusplus
}
#endif

#endif  // SENSE_VOICE_H_

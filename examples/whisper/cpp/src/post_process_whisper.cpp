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

#include "pre_post_common.h"
#include "post_process_whisper.h"

whisper_vocab read_token_info(std::string token_path)
{
    struct whisper_context ctx;
    auto & vocab = ctx.vocab;
    whisper_model_loader loader = {};

    auto fin = std::ifstream(token_path, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s : fail to open '%s'\n", __func__, token_path.c_str());
    }
    loader.context = &fin;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        return fin->eof();
    };

    loader.close = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->close();
    };

    int32_t n_vocab = 0;
    read_safe(&loader, n_vocab);

    std::string word;
    std::vector<char> tmp;

    tmp.reserve(128);

    for (int i = 0; i < n_vocab; i++) {
        uint32_t len;
        read_safe(&loader, len);

        if (len > 0 and i != 50256) {
            tmp.resize(len);
            loader.read(loader.context, &tmp[0], tmp.size()); // read to buffer
            word.assign(tmp.data(), tmp.size());
        } else {
            word = "";
        }

        vocab.token_to_id[word] = i;
        vocab.id_to_token[i] = word;
    }
    fin.eof();
    fin.close();
    n_vocab = 50256;

    if (n_vocab < 51863) {
        // WHISPER_LOG_INFO("%s: adding %d extra tokens\n", __func__, model.hparams.n_vocab - n_vocab);
        for (int i = n_vocab; i < 51863; i++) {
            if (i > vocab.token_beg) {
                word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
            } else if (i == vocab.token_eot) {
                word = "<|endoftext|>";
            } else if (i == vocab.token_sot) {
                word = "<|startoftranscript|>";
            } else if (i == vocab.token_translate) {
                word = "<|translate|>";
            } else if (i == vocab.token_transcribe) {
                word = "<|transcribe|>";
            } else if (i == vocab.token_solm) {
                word = "[_SOLM_]";
            } else if (i == vocab.token_prev) {
                word = "[_PREV_]";
            } else if (i == vocab.token_nosp) {
                word = "[_NOSP_]";
            } else if (i == vocab.token_not) {
                word = "<|notimestamps|>";
            } else if (i == vocab.token_beg) {
                word = "[_BEG_]";
            }
              else if (i == 50258) {
                word= "<|en|>";
            }
              else if (i == 50259) {
                word= "<|zh|>";
            }
              else if (i == 50263) {
                word= "<|ko|>";
            }
              else {
                word = "[_extra_token_" + std::to_string(i) + "]";
            }
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }
    return vocab;
}

std::string do_post_process(int64_t output_id, whisper_vocab vocab)
{
    // std::vector<whisper_token> prompt_init = {50258, 50259, 50359, 50363,2221,13,2326,388,391,307,264,50244,295,264,2808,5359,11,293,321,366,5404,281,2928,702,14943,13,50257};

    std::string text;
    text = vocab.id_to_token.at(output_id).c_str();

    return text;
}
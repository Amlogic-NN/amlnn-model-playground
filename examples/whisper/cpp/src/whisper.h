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
 
#ifndef WHISPER_H
#define WHISPER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include <string>
#include <map>
#include <cstdint>

#ifdef __GNUC__
#    define WHISPER_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define WHISPER_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define WHISPER_DEPRECATED(func, hint) func
#endif

#ifdef WHISPER_SHARED
#    ifdef _WIN32
#        ifdef WHISPER_BUILD
#            define WHISPER_API __declspec(dllexport)
#        else
#            define WHISPER_API __declspec(dllimport)
#        endif
#    else
#        define WHISPER_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define WHISPER_API
#endif

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
#define WHISPER_N_MELS 80
#define WHISPER_N_FRAMES  3000
#define WHISPER_N_SAMPLES 48000

struct whisper_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab = 51864;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    // reference: https://github.com/openai/whisper/blob/248b6cb124225dd263bb9bd32d060b6517e067f8/whisper/tokenizer.py#L334-L349
    id token_eot        = 50256;
    id token_sot        = 50257;
    // task tokens (used only for multilingual models)
    id token_translate  = 50357;
    id token_transcribe = 50358;
    // other special tokens
    id token_solm       = 50359; // [TDRZ] used by tinydiarize models to indicate speaker turn
    id token_prev       = 50360;
    id token_nosp       = 50361;
    id token_not        = 50362; // no timestamps
    id token_beg        = 50363; // begin timestamps

    bool is_multilingual() const {
        return n_vocab >= 51865;
    }

    int num_languages() const {
        return n_vocab - 51765 - (is_multilingual() ? 1 : 0);
    }
};

#ifdef __cplusplus
extern "C" {
#endif

    struct whisper_context;
    struct whisper_state;
    struct whisper_full_params;
    
    typedef int32_t whisper_pos;
    typedef int32_t whisper_token;
    typedef int32_t whisper_seq_id;


    typedef struct whisper_token_data {
        whisper_token id;  // token id
        whisper_token tid; // forced timestamp token id

        float p;           // probability of the token
        float plog;        // log probability of the token
        float pt;          // probability of the timestamp token
        float ptsum;       // sum of probabilities of all timestamp tokens

        // token-level timestamp data
        // do not use if you haven't computed token-level timestamps
        int64_t t0;        // start time of the token
        int64_t t1;        //   end time of the token

        // [EXPERIMENTAL] Token-level timestamps with DTW
        // do not use if you haven't computed token-level timestamps with dtw
        // Roughly corresponds to the moment in audio in which the token was output
        int64_t t_dtw;

        float vlen;        // voice length of the token
    } whisper_token_data;

    typedef struct whisper_model_loader {
        void * context;

        size_t (*read)(void * ctx, void * output, size_t read_size);
        bool    (*eof)(void * ctx);
        void  (*close)(void * ctx);
    } whisper_model_loader;

    // Various functions for loading a ggml whisper model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    WHISPER_API struct whisper_context * whisper_init_from_file_with_params  (const char * path_model,              struct whisper_context_params params);
    WHISPER_API struct whisper_context * whisper_init_from_buffer_with_params(void * buffer, size_t buffer_size,    struct whisper_context_params params);
    WHISPER_API struct whisper_context * whisper_init_with_params            (struct whisper_model_loader * loader, struct whisper_context_params params);

    // These are the same as the above, but the internal state of the context is not allocated automatically
    // It is the responsibility of the caller to allocate the state using whisper_init_state() (#523)
    WHISPER_API struct whisper_context * whisper_init_from_file_with_params_no_state  (const char * path_model,              struct whisper_context_params params);
    WHISPER_API struct whisper_context * whisper_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size,    struct whisper_context_params params);
    WHISPER_API struct whisper_context * whisper_init_with_params_no_state            (struct whisper_model_loader * loader, struct whisper_context_params params);

    WHISPER_API struct whisper_state * whisper_init_state(struct whisper_context * ctx);

    // Given a context, enable use of OpenVINO for encode inference.
    // model_path: Optional path to OpenVINO encoder IR model. If set to nullptr,
    //                      the path will be generated from the ggml model path that was passed
    //                      in to whisper_init_from_file. For example, if 'path_model' was
    //                      "/path/to/ggml-base.en.bin", then OpenVINO IR model path will be
    //                      assumed to be "/path/to/ggml-base.en-encoder-openvino.xml".
    // device: OpenVINO device to run inference on ("CPU", "GPU", etc.)
    // cache_dir: Optional cache directory that can speed up init time, especially for
    //                     GPU, by caching compiled 'blobs' there.
    //                     Set to nullptr if not used.
    // Returns 0 on success. If OpenVINO is not enabled in build, this simply returns 1.
    WHISPER_API int whisper_ctx_init_openvino_encoder(
        struct whisper_context * ctx,
                    const char * model_path,
                    const char * device,
                    const char * cache_dir);

    // Frees all allocated memory
    WHISPER_API void whisper_free      (struct whisper_context * ctx);
    WHISPER_API void whisper_free_state(struct whisper_state * state);
    WHISPER_API void whisper_free_params(struct whisper_full_params * params);
    WHISPER_API void whisper_free_context_params(struct whisper_context_params * params);

    // Convert RAW PCM audio to log mel spectrogram.
    // The resulting spectrogram is stored inside the default state of the provided whisper context.
    // Returns 0 on success
    WHISPER_API int whisper_pcm_to_mel(
            struct whisper_context * ctx,
                       const float * samples,
                               int   n_samples,
                               int   n_threads);

    WHISPER_API int whisper_pcm_to_mel_with_state(
            struct whisper_context * ctx,
              struct whisper_state * state,
                       const float * samples,
                               int   n_samples,
                               int   n_threads);

    // Convert RAW PCM audio to log mel spectrogram but applies a Phase Vocoder to speed up the audio x2.
    // The resulting spectrogram is stored inside the default state of the provided whisper context.
    // Returns 0 on success
    WHISPER_API int whisper_pcm_to_mel_phase_vocoder(
        struct whisper_context * ctx,
                   const float * samples,
                           int   n_samples,
                           int   n_threads);

    WHISPER_API int whisper_pcm_to_mel_phase_vocoder_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
                   const float * samples,
                           int   n_samples,
                           int   n_threads);

    // This can be used to set a custom log mel spectrogram inside the default state of the provided whisper context.
    // Use this instead of whisper_pcm_to_mel() if you want to provide your own log mel spectrogram.
    // n_mel must be 80
    // Returns 0 on success
    WHISPER_API int whisper_set_mel(
            struct whisper_context * ctx,
                       const float * data,
                               int   n_len,
                               int   n_mel);

    WHISPER_API int whisper_set_mel_with_state(
            struct whisper_context * ctx,
              struct whisper_state * state,
                       const float * data,
                               int   n_len,
                               int   n_mel);

    // Run the Whisper encoder on the log mel spectrogram stored inside the default state in the provided whisper context.
    // Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
    // offset can be used to specify the offset of the first frame in the spectrogram.
    // Returns 0 on success
    WHISPER_API int whisper_encode(
            struct whisper_context * ctx,
                               int   offset,
                               int   n_threads);

    WHISPER_API int whisper_encode_with_state(
            struct whisper_context * ctx,
              struct whisper_state * state,
                               int   offset,
                               int   n_threads);

    // Run the Whisper decoder to obtain the logits and probabilities for the next token.
    // Make sure to call whisper_encode() first.
    // tokens + n_tokens is the provided context for the decoder.
    // n_past is the number of tokens to use from previous decoder calls.
    // Returns 0 on success
    // TODO: add support for multiple decoders
    WHISPER_API int whisper_decode(
            struct whisper_context * ctx,
               const whisper_token * tokens,
                               int   n_tokens,
                               int   n_past,
                               int   n_threads);

    WHISPER_API int whisper_decode_with_state(
            struct whisper_context * ctx,
              struct whisper_state * state,
               const whisper_token * tokens,
                               int   n_tokens,
                               int   n_past,
                               int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    WHISPER_API int whisper_tokenize(
            struct whisper_context * ctx,
                        const char * text,
                     whisper_token * tokens,
                               int   n_max_tokens);

    // Return the number of tokens in the provided text
    // Equivalent to: -whisper_tokenize(ctx, text, NULL, 0)
    int whisper_token_count(struct whisper_context * ctx, const char * text);

    // Largest language id (i.e. number of available languages - 1)
    WHISPER_API int whisper_lang_max_id();

    // Return the id of the specified language, returns -1 if not found
    // Examples:
    //   "de" -> 2
    //   "german" -> 2
    WHISPER_API int whisper_lang_id(const char * lang);

    // Return the short string of the specified language id (e.g. 2 -> "de"), returns nullptr if not found
    WHISPER_API const char * whisper_lang_str(int id);

    // Return the short string of the specified language name (e.g. 2 -> "german"), returns nullptr if not found
    WHISPER_API const char * whisper_lang_str_full(int id);

    // Use mel data at offset_ms to try and auto-detect the spoken language
    // Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first
    // Returns the top language id or negative on failure
    // If not null, fills the lang_probs array with the probabilities of all languages
    // The array must be whisper_lang_max_id() + 1 in size
    // ref: https://github.com/openai/whisper/blob/main/whisper/decoding.py#L18-L69
    WHISPER_API int whisper_lang_auto_detect(
            struct whisper_context * ctx,
                               int   offset_ms,
                               int   n_threads,
                             float * lang_probs);

    WHISPER_API int whisper_lang_auto_detect_with_state(
            struct whisper_context * ctx,
              struct whisper_state * state,
                               int   offset_ms,
                               int   n_threads,
                             float * lang_probs);

    WHISPER_API int whisper_n_len           (struct whisper_context * ctx); // mel length
    WHISPER_API int whisper_n_len_from_state(struct whisper_state * state); // mel length
    WHISPER_API int whisper_n_vocab         (struct whisper_context * ctx);
    WHISPER_API int whisper_n_text_ctx      (struct whisper_context * ctx);
    WHISPER_API int whisper_n_audio_ctx     (struct whisper_context * ctx);
    WHISPER_API int whisper_is_multilingual (struct whisper_context * ctx);

    WHISPER_API int whisper_model_n_vocab      (struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_audio_ctx  (struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_audio_state(struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_audio_head (struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_audio_layer(struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_text_ctx   (struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_text_state (struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_text_head  (struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_text_layer (struct whisper_context * ctx);
    WHISPER_API int whisper_model_n_mels       (struct whisper_context * ctx);
    WHISPER_API int whisper_model_ftype        (struct whisper_context * ctx);
    WHISPER_API int whisper_model_type         (struct whisper_context * ctx);

    // Token logits obtained from the last call to whisper_decode()
    // The logits for the last token are stored in the last row
    // Rows: n_tokens
    // Cols: n_vocab
    WHISPER_API float * whisper_get_logits           (struct whisper_context * ctx);
    WHISPER_API float * whisper_get_logits_from_state(struct whisper_state * state);

    // Token Id -> String. Uses the vocabulary in the provided context
    WHISPER_API const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token);
    WHISPER_API const char * whisper_model_type_readable(struct whisper_context * ctx);


    // Special tokens
    WHISPER_API whisper_token whisper_token_eot (struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_sot (struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_solm(struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_prev(struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_nosp(struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_not (struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_beg (struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_lang(struct whisper_context * ctx, int lang_id);

    // Task tokens
    WHISPER_API whisper_token whisper_token_translate (struct whisper_context * ctx);
    WHISPER_API whisper_token whisper_token_transcribe(struct whisper_context * ctx);

    ////////////////////////////////////////////////////////////////////////////

    // Number of generated text segments
    // A segment can be a few words, a sentence, or even a paragraph.
    WHISPER_API int whisper_full_n_segments           (struct whisper_context * ctx);
    WHISPER_API int whisper_full_n_segments_from_state(struct whisper_state * state);

    // Language id associated with the context's default state
    WHISPER_API int whisper_full_lang_id(struct whisper_context * ctx);

    // Language id associated with the provided state
    WHISPER_API int whisper_full_lang_id_from_state(struct whisper_state * state);

    // Get the start and end time of the specified segment
    WHISPER_API int64_t whisper_full_get_segment_t0           (struct whisper_context * ctx, int i_segment);
    WHISPER_API int64_t whisper_full_get_segment_t0_from_state(struct whisper_state * state, int i_segment);

    WHISPER_API int64_t whisper_full_get_segment_t1           (struct whisper_context * ctx, int i_segment);
    WHISPER_API int64_t whisper_full_get_segment_t1_from_state(struct whisper_state * state, int i_segment);

    // Get whether the next segment is predicted as a speaker turn
    WHISPER_API bool whisper_full_get_segment_speaker_turn_next(struct whisper_context * ctx, int i_segment);
    WHISPER_API bool whisper_full_get_segment_speaker_turn_next_from_state(struct whisper_state * state, int i_segment);

    // Get the text of the specified segment
    WHISPER_API const char * whisper_full_get_segment_text           (struct whisper_context * ctx, int i_segment);
    WHISPER_API const char * whisper_full_get_segment_text_from_state(struct whisper_state * state, int i_segment);

    // Get number of tokens in the specified segment
    WHISPER_API int whisper_full_n_tokens           (struct whisper_context * ctx, int i_segment);
    WHISPER_API int whisper_full_n_tokens_from_state(struct whisper_state * state, int i_segment);

    // Get the token text of the specified token in the specified segment
    WHISPER_API const char * whisper_full_get_token_text           (struct whisper_context * ctx, int i_segment, int i_token);
    WHISPER_API const char * whisper_full_get_token_text_from_state(struct whisper_context * ctx, struct whisper_state * state, int i_segment, int i_token);

    WHISPER_API whisper_token whisper_full_get_token_id           (struct whisper_context * ctx, int i_segment, int i_token);
    WHISPER_API whisper_token whisper_full_get_token_id_from_state(struct whisper_state * state, int i_segment, int i_token);

    // Get token data for the specified token in the specified segment
    // This contains probabilities, timestamps, etc.
    WHISPER_API whisper_token_data whisper_full_get_token_data           (struct whisper_context * ctx, int i_segment, int i_token);
    WHISPER_API whisper_token_data whisper_full_get_token_data_from_state(struct whisper_state * state, int i_segment, int i_token);

    // Get the probability of the specified token in the specified segment
    WHISPER_API float whisper_full_get_token_p           (struct whisper_context * ctx, int i_segment, int i_token);
    WHISPER_API float whisper_full_get_token_p_from_state(struct whisper_state * state, int i_segment, int i_token);

    ////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif

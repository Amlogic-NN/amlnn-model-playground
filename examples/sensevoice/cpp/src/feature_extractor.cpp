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

#include "feature_extractor.h"

#include "constants.h"

#include "kaldi-native-fbank/csrc/online-feature.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

std::vector<float> ApplyLfr(const std::vector<float> &frames,
                            int32_t num_frames,
                            int32_t feat_dim) {
    if (num_frames < kLfrWindowSize) {
        return {};
    }

    const int32_t out_num_frames =
        (num_frames - kLfrWindowSize) / kLfrWindowShift + 1;
    const int32_t out_feat_dim = feat_dim * kLfrWindowSize;

    std::vector<float> out(static_cast<size_t>(out_num_frames) * out_feat_dim);

    for (int32_t i = 0; i < out_num_frames; ++i) {
        const float *src = frames.data() + static_cast<size_t>(i * kLfrWindowShift) * feat_dim;
        float *dst = out.data() + static_cast<size_t>(i) * out_feat_dim;
        std::copy(src, src + out_feat_dim, dst);
    }

    return out;
}

std::vector<float> PadOrTruncate(const std::vector<float> &lfr,
                                 int32_t num_frames,
                                 int32_t feat_dim) {
    std::vector<float> out(static_cast<size_t>(kFixedFrames) * feat_dim, 0.f);

    const int32_t copy_frames = std::min(num_frames, kFixedFrames);
    const size_t bytes =
        static_cast<size_t>(copy_frames) * static_cast<size_t>(feat_dim) * sizeof(float);
    std::memcpy(out.data(), lfr.data(), bytes);
    return out;
}

}  // namespace

std::vector<float> ExtractFeatures(const float *samples, int32_t num_samples) {
    if (!samples || num_samples <= 0) {
        return std::vector<float>(static_cast<size_t>(kFixedFrames) * kLfrOutDim, 0.f);
    }

    knf::FbankOptions opts;
    opts.frame_opts.dither = 0.f;
    opts.frame_opts.snip_edges = true;
    opts.frame_opts.samp_freq = kSampleRate;
    opts.frame_opts.frame_shift_ms = 10.f;
    opts.frame_opts.frame_length_ms = 25.f;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.window_type = "hamming";
    opts.mel_opts.num_bins = kFeatureDim;
    opts.mel_opts.high_freq = 0.f;
    opts.mel_opts.low_freq = 20.f;
    opts.mel_opts.is_librosa = false;

    std::vector<float> scaled(static_cast<size_t>(num_samples));
    for (int32_t i = 0; i < num_samples; ++i) {
        scaled[static_cast<size_t>(i)] = samples[i] * 32768.f;
    }

    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(kSampleRate, scaled.data(), num_samples);
    fbank.InputFinished();

    const int32_t num_frames = fbank.NumFramesReady();
    if (num_frames <= 0) {
        return std::vector<float>(static_cast<size_t>(kFixedFrames) * kLfrOutDim, 0.f);
    }

    std::vector<float> frames(static_cast<size_t>(num_frames) * kFeatureDim);
    for (int32_t i = 0; i < num_frames; ++i) {
        const float *frame = fbank.GetFrame(i);
        std::copy(frame, frame + kFeatureDim,
                  frames.data() + static_cast<size_t>(i) * kFeatureDim);
    }

    std::vector<float> lfr = ApplyLfr(frames, num_frames, kFeatureDim);
    if (lfr.empty()) {
        return std::vector<float>(static_cast<size_t>(kFixedFrames) * kLfrOutDim, 0.f);
    }

    const int32_t lfr_frames =
        static_cast<int32_t>(lfr.size() / static_cast<size_t>(kLfrOutDim));
    return PadOrTruncate(lfr, lfr_frames, kLfrOutDim);
}

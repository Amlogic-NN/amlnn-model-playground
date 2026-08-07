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

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <utility>

// Precompute per-output-pixel source contributions for one axis.
static std::vector<std::vector<std::pair<int, float>>> compute_axis_weights(
    int in_size, int out_size)
{
    const double scale = static_cast<double>(out_size) / in_size;
    const double filter_scale = (scale < 1.0) ? (1.0 / scale) : 1.0;
    const double support = 1.0 * filter_scale;

    std::vector<std::vector<std::pair<int, float>>> out(out_size);

    for (int xo = 0; xo < out_size; ++xo) {
        const double center = (xo + 0.5) / scale - 0.5;
        int xmin = static_cast<int>(std::ceil(center - support));
        int xmax = static_cast<int>(std::floor(center + support));

        std::vector<std::pair<int, float>> contribs;
        double weight_sum = 0.0;
        for (int xx = xmin; xx <= xmax; ++xx) {
            const double dist = (xx - center) / filter_scale;
            const double w = 1.0 - std::abs(dist);
            if (w <= 0.0) continue;
            const int clamped = std::max(0, std::min(in_size - 1, xx));
            contribs.emplace_back(clamped, static_cast<float>(w));
            weight_sum += w;
        }
        if (weight_sum > 0.0) {
            for (auto& c : contribs) c.second /= static_cast<float>(weight_sum);
        }
        out[xo] = std::move(contribs);
    }
    return out;
}

// Antialiased bilinear resize. uint8 HWC in -> float [0,1] HWC out.
static std::vector<float> resize_bilinear_antialiased(
    const uint8_t* src, int src_w, int src_h, int channels,
    int dst_w, int dst_h)
{
    auto xw = compute_axis_weights(src_w, dst_w);
    auto yw = compute_axis_weights(src_h, dst_h);

    std::vector<float> tmp(static_cast<size_t>(dst_w) * src_h * channels);
    for (int y = 0; y < src_h; ++y) {
        const uint8_t* row = src + static_cast<size_t>(y) * src_w * channels;
        for (int x = 0; x < dst_w; ++x) {
            float* out = &tmp[(static_cast<size_t>(y) * dst_w + x) * channels];
            for (int c = 0; c < channels; ++c) out[c] = 0.0f;
            for (const auto& pr : xw[x]) {
                const uint8_t* p = row + static_cast<size_t>(pr.first) * channels;
                const float w = pr.second;
                for (int c = 0; c < channels; ++c) out[c] += static_cast<float>(p[c]) * w;
            }
        }
    }

    std::vector<float> dst(static_cast<size_t>(dst_w) * dst_h * channels);
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            float* out = &dst[(static_cast<size_t>(y) * dst_w + x) * channels];
            for (int c = 0; c < channels; ++c) out[c] = 0.0f;
            for (const auto& pr : yw[y]) {
                const float* p = &tmp[(static_cast<size_t>(pr.first) * dst_w + x) * channels];
                const float w = pr.second;
                for (int c = 0; c < channels; ++c) out[c] += p[c] * w;
            }
            for (int c = 0; c < channels; ++c) out[c] /= 255.0f;
        }
    }
    return dst;
}

// Preprocess raw RGB image to match open_clip MobileCLIP-S2 preprocess:
//   resize shortest edge -> 256 (keep aspect ratio)
//   center crop -> 256x256
//   uint8 [0,255] -> float [0,1]
//   mean=(0,0,0), std=(1,1,1) (no extra normalization)
// Output layout: CHW.
std::vector<float> preprocess_image_impl(const std::vector<uint8_t>& rgb_data, int width, int height) {
    const int target_size = 256;

    // 1) Resize shortest edge to target_size.
    float scale = static_cast<float>(target_size) / std::min(width, height);
    int new_w = std::max(target_size, static_cast<int>(std::round(width * scale)));
    int new_h = std::max(target_size, static_cast<int>(std::round(height * scale)));

    std::vector<float> resized = resize_bilinear_antialiased(rgb_data.data(), width, height, 3, new_w, new_h);

    // 2) Center crop to target_size x target_size, HWC -> CHW.
    int left = (new_w - target_size) / 2;
    int top  = (new_h - target_size) / 2;

    std::vector<float> chw(static_cast<size_t>(target_size) * target_size * 3);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_size; ++h) {
            for (int w = 0; w < target_size; ++w) {
                chw[c * target_size * target_size + h * target_size + w] =
                    resized[((h + top) * new_w + (w + left)) * 3 + c];
            }
        }
    }

    return chw;
}

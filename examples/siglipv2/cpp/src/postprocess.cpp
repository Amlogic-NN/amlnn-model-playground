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

#include "postprocess.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Bilinear interpolation scaling
static std::vector<float> resize_bilinear(
    const unsigned char *src, int src_w, int src_h, int channels,
    int dst_w, int dst_h)
{
    std::vector<float> dst(dst_w * dst_h * channels);

    for (int y = 0; y < dst_h; y++)
    {
        float fy = (y + 0.5f) * src_h / dst_h - 0.5f;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(src_h - 1, y0 + 1);
        float wy = fy - y0;

        for (int x = 0; x < dst_w; x++)
        {
            float fx = (x + 0.5f) * src_w / dst_w - 0.5f;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(src_w - 1, x0 + 1);
            float wx = fx - x0;

            for (int c = 0; c < channels; c++)
            {
                float v00 = src[(y0 * src_w + x0) * channels + c];
                float v01 = src[(y0 * src_w + x1) * channels + c];
                float v10 = src[(y1 * src_w + x0) * channels + c];
                float v11 = src[(y1 * src_w + x1) * channels + c];
                float v0 = v00 * (1 - wx) + v01 * wx;
                float v1 = v10 * (1 - wx) + v11 * wx;
                float v = v0 * (1 - wy) + v1 * wy;

                // Keep scaled down to [0, 1] internally for the moment
                dst[(y * dst_w + x) * channels + c] = v / 255.0f;
            }
        }
    }
    return dst;
}

std::vector<float> preprocess_image(const std::string &image_path, int target_w, int target_h)
{
    int width, height, channels;
    unsigned char *img = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!img)
    {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return {};
    }

    // SigLIP uses direct resize, no center cropping
    std::vector<float> resized = resize_bilinear(img, width, height, 3, target_w, target_h);
    stbi_image_free(img);

    // SigLIP Normalization
    // Python equivalent: (image - 127.5) / 127.5
    // Since our `resized` is currently mapped [0, 1], we map it to match:
    // (val * 255.0 - 127.5) / 127.5 = (val * 2.0) - 1.0
    for (size_t i = 0; i < resized.size(); i++)
    {
        resized[i] = resized[i] * 2.0f - 1.0f;
    }

    // Returns NHWC
    return resized;
}

// ==================== Post Processing ====================

// NOTE: SigLIP does *NOT* L2 Normalize embeddings! We just compute the raw dot product.
float compute_similarity(const std::vector<float> &a, const std::vector<float> &b)
{
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
    }
    return dot;
}

float sigmoid(float x)
{
    x = std::max(-80.0f, std::min(80.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}
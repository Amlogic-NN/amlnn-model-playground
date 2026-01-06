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
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include "model_invoke.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// bilinear interpolation scaling
std::vector<float> resize_bilinear(
    const unsigned char* src, int src_w, int src_h, int channels,
    int dst_w, int dst_h)
{
    std::vector<float> dst(dst_w * dst_h * channels);

    for (int y = 0; y < dst_h; y++) {
        float fy = (y + 0.5f) * src_h / dst_h - 0.5f;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(src_h - 1, y0 + 1);
        float wy = fy - y0;

        for (int x = 0; x < dst_w; x++) {
            float fx = (x + 0.5f) * src_w / dst_w - 0.5f;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(src_w - 1, x0 + 1);
            float wx = fx - x0;

            for (int c = 0; c < channels; c++) {
                float v00 = src[(y0 * src_w + x0) * channels + c];
                float v01 = src[(y0 * src_w + x1) * channels + c];
                float v10 = src[(y1 * src_w + x0) * channels + c];
                float v11 = src[(y1 * src_w + x1) * channels + c];
                float v0 = v00 * (1 - wx) + v01 * wx;
                float v1 = v10 * (1 - wx) + v11 * wx;
                float v = v0 * (1 - wy) + v1 * wy;
                dst[(y * dst_w + x) * channels + c] = v / 255.0f;
            }
        }
    }
    return dst;
}

std::vector<float> preprocess_image(const std::string& image_path) {
    int width, height, channels;
    unsigned char* img = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return {};
    }

    const int target_size = 224;

    // scale the shorter side
    float scale = (float)target_size / std::min(width, height);
    int new_w = std::round(width * scale);
    int new_h = std::round(height * scale);

    // resize
    std::vector<float> resized = resize_bilinear(img, width, height, 3, new_w, new_h);

    // center crop
    int left = (new_w - target_size) / 2;
    int top  = (new_h - target_size) / 2;

    std::vector<float> cropped(target_size * target_size * 3);
    for (int h = 0; h < target_size; h++) {
        for (int w = 0; w < target_size; w++) {
            for (int c = 0; c < 3; c++) {
                cropped[(h * target_size + w) * 3 + c] =
                    resized[((h + top) * new_w + (w + left)) * 3 + c];
            }
        }
    }

    stbi_image_free(img);

    // normalization (CLIP)
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

    for (int i = 0; i < target_size * target_size; i++) {
        for (int c = 0; c < 3; c++) {
            cropped[i * 3 + c] = (cropped[i * 3 + c] - mean[c]) / std[c];
        }
    }

    // get NHWC
    return cropped;
}

float post_process(const float* a, const std::vector<float>& b) {
    float dot = 0.0f, scale = 100.00000762939453f;
    for (size_t i = 0; i < b.size(); ++i) {
        dot += a[i] * b[i];
    }
    dot *= scale;
    return dot;
}

float post_process(const int8_t* a, const std::vector<float>& b) {
    float dot = 0.0f, scale = 100.00000762939453f;
    for (size_t i = 0; i < b.size(); ++i) {
        dot += (a[i] - 66) * b[i];
    }
    dot *= scale;
    return dot;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> result(logits.size());

    // numerical stability: subtract the maximum value first.
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - max_logit);
        sum_exp += result[i];
    }

    for (float& val : result) {
        val /= sum_exp;
    }

    return result;
}

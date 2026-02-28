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

#ifndef CLIP_PROCESS_H
#define CLIP_PROCESS_H

#include <string>
#include <vector>
#include <cstdint>

// ==================== Model Invoke ====================

// Initialize network from file
void* init_network_file(const char *model_path);

// Run image model inference
std::vector<float> run_image_model(void* context, const std::vector<float>& input_data);

// Run text model inference
std::vector<float> run_text_model(void* context, const std::vector<int64_t>& input_ids);

// Destroy network
int destroy_network(void *qcontext);

// ==================== Pre/Post Processing ====================

// Image preprocessing
std::vector<float> preprocess_image(const std::string& image_path);

// L2 normalize
std::vector<float> l2_normalize(const std::vector<float>& vec);

// Softmax
std::vector<float> softmax(const std::vector<float>& logits);

// Compute cosine similarity
float compute_similarity(const std::vector<float>& a, const std::vector<float>& b, float scale = 100.0f);

#endif // CLIP_PROCESS_H


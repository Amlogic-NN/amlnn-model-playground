/*
 * Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODEL_INVOKE_H
#define MODEL_INVOKE_H

#include <string>
#include <vector>
#include <map>

void* init_network_file(const char *model_path);
std::vector<std::string> process_image_dir(void *context_model, const std::string& json_path, const std::string& base_dir = "", const std::string& json_filename = "");
int destroy_network(void *qcontext);

#endif // MODEL_INVOKE_H


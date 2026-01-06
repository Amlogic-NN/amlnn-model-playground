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

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstdlib>

#include "model_invoke.h"
#include "nn_sdk.h"
#include "json.hpp"
#include <filesystem>
#include <regex>

using json = nlohmann::ordered_json;
namespace fs = std::__fs::filesystem;

struct DMAConfig {
    bool use_dma = true;
    bool malloc_buffer_once = true;
};

DMAConfig context_model;

///////////////////////////////////////////////////////////

aml_memory_config_t mem_config_context_model;
aml_memory_data_t mem_data_context_model;

std::vector<float> preprocess_image(const std::string& image_path);
float post_process(const float* a, const std::vector<float>& b);

void* init_network_file(const char *model_path)
{
    void *qcontext = NULL;
    aml_config config;

    memset(&config, 0, sizeof(aml_config));
    config.nbgType = NN_ADLA_FILE;
    config.path = model_path;
    config.modelType = ADLA_LOADABLE;
    config.typeSize = sizeof(aml_config);

    /* set omp, If you are considering high CPU usage during operation,
       you can turn off this api, set_openmp_opt_flag = false */
    aml_openmp_opt_t openmp_opt[] = 
    {
        {
           .operator_type = AML_Unknown,
           .enable_openmp = true,
           .involve_all_ops = true,
           .openmp_num = 2,
        },
    };
    config.forward_ctrl.softop_info.set_openmp_opt_flag = true;
    config.forward_ctrl.softop_info.openmp_opt_num = sizeof(openmp_opt) / sizeof(aml_openmp_opt_t);
    config.forward_ctrl.softop_info.openmp_opt = openmp_opt;

    /* set neon */
    aml_neon_opt_t neon_opt[] = 
    {
        {
           .operator_type = AML_Unknown,
           .enable_neon = true,
           .involve_all_ops = true,
        },
    };
    config.forward_ctrl.softop_info.set_neon_opt_flag = true;
    config.forward_ctrl.softop_info.neon_opt_num = sizeof(neon_opt) / sizeof(aml_neon_opt_t);
    config.forward_ctrl.softop_info.neon_opt = neon_opt;

    qcontext = aml_module_create(&config);
    if (NULL == qcontext)
    {
        printf("aml_module_create fail.\n");
        return NULL;
    }

    return qcontext;
}

float* run_network(void *qcontext, std::vector<float> input_ids, const std::string image_type)
{
    int ret = 0;
    nn_input inData;

    nn_output *outdata = NULL;
    aml_output_config_t outconfig;

    inData.input_index = 0;
    inData.info.input_format = AML_INPUT_DEFAULT;
    inData.size = input_ids.size() * sizeof(float);

    if (context_model.use_dma) {
        if (context_model.malloc_buffer_once) {
            mem_config_context_model.cache_type = AML_WITH_CACHE;
            mem_config_context_model.memory_type = AML_VIRTUAL_ADDR;
            mem_config_context_model.direction = AML_MEM_DIRECTION_READ_WRITE;
            mem_config_context_model.index = 0;
            mem_config_context_model.mem_size = inData.size;
            aml_util_mallocBuffer(qcontext, &mem_config_context_model, &mem_data_context_model);
            aml_util_swapExternalInputBuffer(qcontext, &mem_config_context_model, &mem_data_context_model);
        }

        inData.input_type = INPUT_DMA_DATA;
        memcpy(mem_data_context_model.viraddr, input_ids.data(), mem_config_context_model.mem_size);
        inData.input = NULL;
    } else {
        inData.input = reinterpret_cast<unsigned char*>(input_ids.data());
        inData.input_type = BINARY_RAW_DATA;

        ret = aml_module_input_set(qcontext, &inData);
        if (ret)
        {
            printf("aml_module_input_set fail.\n");
        }
    }
    context_model.malloc_buffer_once = false;

    memset(&outconfig, 0, sizeof(aml_output_config_t));

    if (context_model.use_dma) {
        outconfig.format = AML_OUTDATA_DMA;
    } else {
        outconfig.format = AML_OUTDATA_RAW;
    }
    outconfig.typeSize = sizeof(aml_output_config_t);
    outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);

    return reinterpret_cast<float*>(outdata->out[0].buf);
}

int extract_index(const std::string& filename) {
    std::regex pattern(R"(test_\w+_(\d+)\.jpg)");
    std::smatch match;
    if (std::regex_match(filename, match, pattern)) {
        return std::stoi(match[1]);
    }
    return -1;
}

std::vector<std::string> process_image_dir(
    void* context_model,
    const std::string& image_dir_path,
    const std::string& base_dir,
    const std::string& json_filename)
{
    std::vector<std::string> results;
    std::regex file_pattern(R"(test_(\w+)_\d+\.jpg)");
    
    // Get base_dir from parameter, environment variable, or use default
    std::string actual_base_dir = base_dir;
    if (actual_base_dir.empty()) {
        const char* env_base_dir = std::getenv("CLIP_BASE_DIR");
        if (env_base_dir != nullptr) {
            actual_base_dir = env_base_dir;
        } else {
            actual_base_dir = "./demo_data/clip_datasets/";
        }
    }
    
    // Ensure base_dir ends with '/'
    if (!actual_base_dir.empty() && actual_base_dir.back() != '/') {
        actual_base_dir += "/";
    }
    
    // Get json_filename from parameter, environment variable, or use default
    std::string actual_json_filename = json_filename;
    if (actual_json_filename.empty()) {
        const char* env_json_filename = std::getenv("CLIP_JSON_FILENAME");
        if (env_json_filename != nullptr) {
            actual_json_filename = env_json_filename;
        } else {
            actual_json_filename = "clip_text_res.json";
        }
    }

    // storing qualified paths
    std::vector<fs::directory_entry> matched_files;

    // collect all relevant img.
    for (const auto& entry : fs::directory_iterator(image_dir_path)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, file_pattern)) {
            matched_files.push_back(entry);
        }
    }

    // use index sort, test_type_index.jpg
    std::sort(matched_files.begin(), matched_files.end(),
        [](const fs::directory_entry& a, const fs::directory_entry& b) {
            return extract_index(a.path().filename().string()) <
                   extract_index(b.path().filename().string());
        });

    for (const auto& entry : matched_files) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(filename, match, file_pattern)) continue;

        std::string name = match[1];

        std::vector<float> input_data = preprocess_image(entry.path().string());
        float* model_output = run_network(context_model, input_data, name);

        float max_sim = -std::numeric_limits<float>::infinity();
        std::string best_key, best_id;

        // Iterate through all directories to find the directory containing the name
        for (const auto& dir_entry : fs::directory_iterator(actual_base_dir)) {
            if (!dir_entry.is_directory()) continue;

            std::string folder_name = dir_entry.path().filename().string();
            if (folder_name.find(name) == std::string::npos) continue;

            std::string vit_res_path = actual_base_dir + folder_name + "/" + actual_json_filename;
            std::ifstream vit_in(vit_res_path);
            if (!vit_in.is_open()) {
                printf("unopen: %s\n", vit_res_path.c_str());
                continue;
            }

            json vit_json;
            vit_in >> vit_json;

            for (auto it = vit_json.begin(); it != vit_json.end(); ++it) {
                const std::string& key = it.key();
                const std::vector<float> vec = it.value().get<std::vector<float>>();
                float sim = post_process(model_output, vec);
                // printf("sim: %.4f\n", sim);
                if (sim > max_sim) {
                    max_sim = sim;
                    best_key = key;
                    best_id = folder_name;
                }
            }
        }

        if (!best_key.empty() && !best_id.empty()) {
            std::string best_path = actual_base_dir + best_id + "/";
            results.push_back(best_path);
            printf("\nProcessing images: %s, datasets img path: %s\n", filename.c_str(), best_path.c_str());
            // printf("最相似图片: %s 相似度: %.4f\n", best_path.c_str(), max_sim);    // for debug
        }
    }

    return results;
}


int destroy_network(void *qcontext)
{
    int ret = 0;

    /* free model 
       model.use_dma = true
       model.malloc_buffer_once = false
    */
    if (context_model.use_dma && mem_config_context_model.mem_size != 0) {
        ret = aml_util_freeBuffer(qcontext, &mem_config_context_model, &mem_data_context_model);
        if (ret)
        {
            std::cout << "aml_util_freeBuffer fail." << std::endl;
        }
    }
    context_model.use_dma = false;

    ret = aml_module_destroy(qcontext);
    if (ret)
    {
        printf("aml_module_destroy fail.\n");
        return -1;
    }

    return ret;
}
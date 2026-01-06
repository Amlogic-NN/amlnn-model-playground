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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "model_invoke.h"

#define BILLION 1000000000

struct Get_Times
{
    uint64_t init_start_time, init_end_time, init_total_time;
    uint64_t preProcess_start_time, preProcess_end_time, preProcess_total_time;
    uint64_t invoke_start_time, invoke_end_time, invoke_total_time;
    uint64_t postProcess_start_time, postProcess_end_time, postProcess_total_time;
    uint64_t total_time;
    std::vector<uint64_t> total_time_group;
};

static uint64_t get_time_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
}

int main(int argc, char ** argv)
{
    Get_Times model_time;

    std::vector<float> input_data_fir;
    float* model_output_data;
 
    int ret = 0;
    int max_index = 0;
    
    if (argc < 2) {
        printf("Usage: %s <model_path> [base_dir] [json_filename]\n", argv[0]);
        printf("  model_path:   Path to the model file\n");
        printf("  base_dir:     Base directory for clip datasets (optional, can also use CLIP_BASE_DIR env var)\n");
        printf("  json_filename: JSON filename in each dataset folder (optional, can also use CLIP_JSON_FILENAME env var, default: clip_text_res.json)\n");
        return -1;
    }
    
    char* model_path_encoder = argv[1];
    std::string base_dir = (argc >= 3) ? argv[2] : "";
    std::string json_filename = (argc >= 4) ? argv[3] : "";
    void *context_model = NULL;

    model_time.init_start_time = get_time_count();
    context_model = init_network_file(model_path_encoder);
    model_time.init_end_time = get_time_count();

    if (context_model == NULL)
    {
        printf("init_network [context_model] fail.\n");
        return -1;
    }

    if (getenv("GET_TIME"))
    {
        model_time.init_total_time = (model_time.init_end_time - model_time.init_start_time) / 1000000;
        std::cout << "init_model_total time : " << model_time.init_total_time << "ms" << std::endl;
    }

    while (true)
    {
        std::string json_path;

        printf("\nPlease enter the JPG image path (enter exit to quit):\n");
        std::getline(std::cin, json_path);
        if (json_path == "exit") break;
        if (json_path.empty()) {
            printf("The path cannot be empty.\n");
            continue;
        }
        std::vector<std::string> out_str_path = process_image_dir(context_model, json_path, base_dir, json_filename);

        for (int i = 0; i < out_str_path.size(); i++)
        {
            std::cout << "Index[" << i << "] : " << out_str_path[i] << std::endl;
        }
    }

    ret = destroy_network(context_model);
    if (ret != 0)
    {
        printf("destroy_network [context_model] fail.\n");
        return -1;
    }

    return ret;
}
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
#include <cmath>
#include <cstdlib>

#include "clip_process.h"
#include "nn_sdk.h"

// Global DMA config for models
static aml_memory_config_t vision_mem_config;
static aml_memory_data_t vision_mem_data;
static void* vision_context_flag = nullptr;

static aml_memory_config_t text_mem_config;
static aml_memory_data_t text_mem_data;
static void* text_context_flag = nullptr;

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

std::vector<float> run_vision_model(void* qcontext, const std::vector<float>& input_data)
{
    int ret = 0;
    nn_input inData;
    nn_output *outdata = NULL;
    aml_output_config_t outconfig;

    inData.input_index = 0;
    inData.info.input_format = AML_INPUT_DEFAULT;
    inData.size = input_data.size() * sizeof(float);

    // Use DMA
    if (!vision_context_flag) {
        vision_mem_config.cache_type = AML_WITH_CACHE;
        vision_mem_config.memory_type = AML_VIRTUAL_ADDR;
        vision_mem_config.direction = AML_MEM_DIRECTION_READ_WRITE;
        vision_mem_config.index = 0;
        vision_mem_config.mem_size = inData.size;
        aml_util_mallocBuffer(qcontext, &vision_mem_config, &vision_mem_data);
        aml_util_swapExternalInputBuffer(qcontext, &vision_mem_config, &vision_mem_data);
        vision_context_flag = qcontext;
    }

    inData.input_type = INPUT_DMA_DATA;
    memcpy(vision_mem_data.viraddr, input_data.data(), vision_mem_config.mem_size);
    inData.input = NULL;

    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.format = AML_OUTDATA_DMA;
    outconfig.typeSize = sizeof(aml_output_config_t);
    outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);

    if (outdata == NULL || outdata->out[0].buf == NULL) {
        printf("Vision model inference failed.\n");
        return {};
    }

    // Copy output to vector
    size_t output_size = outdata->out[0].size / sizeof(float);
    float* output_ptr = reinterpret_cast<float*>(outdata->out[0].buf);
    std::vector<float> result(output_ptr, output_ptr + output_size);

    return result;
}

std::vector<float> run_text_model(void* qcontext, const std::vector<int64_t>& input_ids)
{
    int ret = 0;
    nn_input inData;
    nn_output *outdata = NULL;
    aml_output_config_t outconfig;

    inData.input_index = 0;
    inData.info.input_format = AML_INPUT_DEFAULT;
    inData.size = input_ids.size() * sizeof(int64_t);

    // Use DMA
    if (!text_context_flag) {
        text_mem_config.cache_type = AML_WITH_CACHE;
        text_mem_config.memory_type = AML_VIRTUAL_ADDR;
        text_mem_config.direction = AML_MEM_DIRECTION_READ_WRITE;
        text_mem_config.index = 0;
        text_mem_config.mem_size = inData.size;
        aml_util_mallocBuffer(qcontext, &text_mem_config, &text_mem_data);
        aml_util_swapExternalInputBuffer(qcontext, &text_mem_config, &text_mem_data);
        text_context_flag = qcontext;
    }

    inData.input_type = INPUT_DMA_DATA;
    memcpy(text_mem_data.viraddr, input_ids.data(), text_mem_config.mem_size);
    inData.input = NULL;

    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.format = AML_OUTDATA_DMA;
    outconfig.typeSize = sizeof(aml_output_config_t);
    outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);

    if (outdata == NULL || outdata->out[0].buf == NULL) {
        printf("Text model inference failed.\n");
        return {};
    }

    // Copy output to vector
    size_t output_size = outdata->out[0].size / sizeof(float);
    float* output_ptr = reinterpret_cast<float*>(outdata->out[0].buf);
    std::vector<float> result(output_ptr, output_ptr + output_size);

    return result;
}

int destroy_network(void *qcontext)
{
    int ret = 0;

    if (vision_context_flag == qcontext) {
        printf("Free vision model memory.\n");
        aml_util_freeBuffer(qcontext, &vision_mem_config, &vision_mem_data);
        vision_context_flag = nullptr;
    } else if (text_context_flag == qcontext) {
        printf("Free text model memory.\n");
        aml_util_freeBuffer(qcontext, &text_mem_config, &text_mem_data);
        text_context_flag = nullptr;
    } else {
        printf("Free network failed: context not found.\n");
        return -1;
    }

    ret = aml_module_destroy(qcontext);
    if (ret)
    {
        printf("Free network failed: destroy failed.\n");
        return -1;
    }

    return ret;
}

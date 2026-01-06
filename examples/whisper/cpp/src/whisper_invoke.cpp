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
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <algorithm>

#include "nn_sdk.h"
#include "whisper.h"
#include "whisper_invoke.h"

struct DMAConfig {
    bool use_dma = true;
    bool malloc_buffer_once = true;
};

DMAConfig encoder, decoder;

bool is_finish = false;

static int decoder_input_1_size = 2;         /* init decoder input_1 size*/
whisper_vocab vocab_out_init;

///////////////////////////////////////////////////////////

#define TIKTOKEN_ID_STOP 50256
#define INPUT_SHAPE 48
#define decoder_model_inputs_size 2

aml_memory_config_t mem_config_encoder;
aml_memory_data_t mem_data_encoder;

aml_memory_config_t mem_config[decoder_model_inputs_size];
aml_memory_data_t mem_data[decoder_model_inputs_size];

whisper_vocab read_token_info(std::string token_path);

void* init_network_file(const char *model_path)
{
    void *qcontext = NULL;
    aml_config config;
    static bool is_worker_initialized = false;

    if (!is_worker_initialized) {
        vocab_out_init = read_token_info("./data_bin/tokenizer_info.bin");
        is_worker_initialized = true;
    }

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

bool is_finish_end() {
    return is_finish;
}

std::vector<float> run_network_encoder_process(void *qcontext, std::vector<float> input_ids)
{
    int ret = 0;
    nn_input inData;
    size_t outputData_size;

    nn_output *outdata = NULL;
    aml_output_config_t outconfig;

    is_finish = false; /* init is_finish -> false */
    inData.input_index = 0;
    inData.info.input_format = AML_INPUT_DEFAULT;
    inData.size = input_ids.size() * sizeof(float);    /* INPUT_SHAPE --->>> input_ids.size() */

    if (encoder.use_dma) {
        if (encoder.malloc_buffer_once) {
            mem_config_encoder.cache_type = AML_WITH_CACHE;
            mem_config_encoder.memory_type = AML_VIRTUAL_ADDR;
            mem_config_encoder.direction = AML_MEM_DIRECTION_READ_WRITE;
            mem_config_encoder.index = 0;
            mem_config_encoder.mem_size = inData.size;
            aml_util_mallocBuffer(qcontext, &mem_config_encoder, &mem_data_encoder);
            aml_util_swapExternalInputBuffer(qcontext, &mem_config_encoder, &mem_data_encoder);
        }

        inData.input_type = INPUT_DMA_DATA;
        memcpy(mem_data_encoder.viraddr, input_ids.data(), mem_config_encoder.mem_size);
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
    encoder.malloc_buffer_once = false;

    memset(&outconfig, 0, sizeof(aml_output_config_t));

    if (encoder.use_dma) {
        outconfig.format = AML_OUTDATA_DMA;
    } else {
        outconfig.format = AML_OUTDATA_RAW;
    }
    outconfig.typeSize = sizeof(aml_output_config_t);
    outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);

    outputData_size = outdata->out[0].size / sizeof(float);
    std::vector<float> buf_data(reinterpret_cast<float*>(outdata->out[0].buf), reinterpret_cast<float*>(outdata->out[0].buf) + outputData_size);

    return buf_data;
}

nn_output* run_network_decoder_process(void *qcontext, Input_Decoder* input_data)
{
    int ret = 0;
    nn_input inData;

    nn_output *outdata = NULL;
    aml_output_config_t outconfig;

    for (int i = 0; i < decoder_model_inputs_size; i++)
    {
        inData.input_index = i;
        inData.info.input_format = AML_INPUT_DEFAULT;

        inData.size = i == 0 ? input_data->input_0_size * sizeof(float) : input_data->input_1_size * sizeof(int64_t);

        if (decoder.use_dma) {
            if (decoder.malloc_buffer_once) {
                mem_config[i].index = i;
                mem_config[i].mem_size = inData.size;
                mem_config[i].cache_type = AML_WITH_CACHE;
                mem_config[i].memory_type = AML_VIRTUAL_ADDR;
                mem_config[i].direction = AML_MEM_DIRECTION_READ_WRITE;
                aml_util_mallocBuffer(qcontext, &mem_config[i], &mem_data[i]);
                aml_util_swapExternalInputBuffer(qcontext, &mem_config[i], &mem_data[i]);
            }

            inData.input_type = INPUT_DMA_DATA;
            memcpy(mem_data[i].viraddr, i == 0 ? static_cast<const void*>(input_data->input_0) : 
                    static_cast<const void*>(input_data->input_1), mem_config[i].mem_size);
            inData.input = NULL;
        } else {
            inData.input = i == 0 ? reinterpret_cast<unsigned char*>(const_cast<float*>(input_data->input_0)) : 
                    reinterpret_cast<unsigned char*>(const_cast<int64_t*>(input_data->input_1));
            inData.input_type = BINARY_RAW_DATA;

            ret = aml_module_input_set(qcontext, &inData);
            if (ret)
            {
                printf("aml_module_input_set fail.\n");
            }
        }
    }
    decoder.malloc_buffer_once = false;

    memset(&outconfig, 0, sizeof(aml_output_config_t));

    if (decoder.use_dma) {
        outconfig.format = AML_OUTDATA_DMA;
    } else {
        outconfig.format = AML_OUTDATA_RAW;
    }
    outconfig.typeSize = sizeof(aml_output_config_t);

    outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);

    return outdata;
}

std::string run_network_decoder(void *qcontext_sec, Input_Decoder* input_data)
{
    int ret = 0;
    int max_index = 0;
    std::string out;
    size_t id_shape, begin_count, last_count;

    nn_output* buf_data_sec;

    buf_data_sec = run_network_decoder_process(qcontext_sec, input_data);

    float* buf_data = reinterpret_cast<float*>(buf_data_sec->out[0].buf);

    id_shape = decoder_input_1_size;

    begin_count = (id_shape - 1) * 51864;    // why id_shape -1? output[0] shape [1, 64, 51864], save [id_shape - 1] group data
    last_count = id_shape * 51864 - 1;

    // get max_valus and max_index
    auto max_it = std::max_element(buf_data + begin_count, buf_data + last_count);
    max_index = std::distance(buf_data + begin_count, max_it);

    input_data->input_1[id_shape] = max_index;

    if (max_index == TIKTOKEN_ID_STOP || id_shape >= INPUT_SHAPE) {
        is_finish = true;
        if (max_index != TIKTOKEN_ID_STOP)
            out = vocab_out_init.id_to_token.at(max_index).c_str();
        decoder_input_1_size = 2;
    }
    else {
        out = vocab_out_init.id_to_token.at(max_index).c_str();
        decoder_input_1_size++;
    }

    return out;
}

int destroy_network(void *qcontext)
{
    int ret = 0;

    /* free encoder 
       encoder.use_dma = true
       encoder.malloc_buffer_once = false
    */
    if (encoder.use_dma && mem_config_encoder.mem_size != 0) {
        ret = aml_util_freeBuffer(qcontext, &mem_config_encoder, &mem_data_encoder);
        if (ret)
        {
            std::cout << "aml_util_freeBuffer fail." << std::endl;
        }
    }
    encoder.use_dma = false;

    /* free decoder 
       first use destroy_network, decoder.malloc_buffer_once is false,
       and set decoder.malloc_buffer_once is true
    */
    if (decoder.malloc_buffer_once && mem_config[0].mem_size != 0) {
        for (int i = 0; i < decoder_model_inputs_size; i++)
        {
            ret = aml_util_freeBuffer(qcontext, &mem_config[i], &mem_data[i]);
            if (ret)
            {
                std::cout << "aml_util_freeBuffer fail." << std::endl;
            }
        }
    }
    decoder.malloc_buffer_once = true;

    ret = aml_module_destroy(qcontext);
    if (ret)
    {
        printf("aml_module_destroy fail.\n");
        return -1;
    }

    return ret;
}
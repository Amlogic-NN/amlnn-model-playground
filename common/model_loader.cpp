// -------------------------------------------------------------------------
// Exposed Functions
// -------------------------------------------------------------------------

#include "model_loader.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>

#define LOGE(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)

void* init_network(const char* model_path) {
    void* qcontext = NULL;
    aml_config config;
    memset(&config, 0, sizeof(aml_config));
    config.modelType = ADLA_LOADABLE;
    config.typeSize = sizeof(aml_config);
    config.nbgType = NN_ADLA_FILE;
    config.path = model_path;

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
    if (NULL == qcontext) {
        LOGE("aml_module_create fail for %s", model_path);
        return NULL;
    }
    return qcontext;
}

int uninit_network(void* qcontext) {
    int ret = aml_module_destroy(qcontext);
    if (ret) {
        LOGE("aml_module_destroy fail.");
        return -1;
    }   
    return ret;
}

void* run_network(void* qcontext, std::vector<std::tuple<cv::Mat, float, std::tuple<int, int>>> input_tuples) {
    for (size_t i = 0; i < input_tuples.size(); ++i) {
        cv::Mat process_img = std::get<0>(input_tuples[i]);
        unsigned char* rawdata = process_img.data;

        nn_input inData;
        memset(&inData, 0, sizeof(nn_input));
        inData.input_type = BINARY_RAW_DATA;
        inData.input = rawdata;
        inData.input_index = i;
        inData.size = process_img.rows * process_img.cols * process_img.channels() * sizeof(float);

        int ret = aml_module_input_set(qcontext, &inData);
        if (ret) {
            LOGE("aml_module_input_set fail for index %zu. Ret=%d", i, ret);
            return NULL;
        }
    }

    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.format = AML_OUTDATA_FLOAT32;
    return aml_module_output_get(qcontext, outconfig);
}

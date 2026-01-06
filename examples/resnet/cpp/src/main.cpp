#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "nn_sdk.h"
#include "postprocess.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir> <labels.txt>\n";
        return 0;
    }

    auto labels = load_labels(argv[3]);

    aml_config cfg{};
    cfg.typeSize = sizeof(cfg);
    cfg.modelType = ADLA_LOADABLE;
    cfg.nbgType = NN_ADLA_FILE;
    cfg.path = argv[1];
    void* ctx = aml_module_create(&cfg);
    if (!ctx) return -1;

    std::vector<float> input_buffer(kInputW * kInputH * 3);

    for (auto& it : fs::directory_iterator(argv[2])) {
        cv::Mat img = cv::imread(it.path().string());
        if (img.empty()) continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing: " << it.path().filename() << std::endl;

        preprocess(img, input_buffer.data());

        nn_input in{};
        in.typeSize = sizeof(in);
        in.input_type = BINARY_RAW_DATA;
        in.input = (unsigned char*)input_buffer.data();
        in.size = input_buffer.size() * sizeof(float);
        in.info.valid = 1;
        in.info.input_format = AML_INPUT_MODEL_NHWC; 
        in.info.input_data_type = AML_INPUT_FP32;
        aml_module_input_set(ctx, &in);

        aml_output_config_t outcfg{};
        outcfg.typeSize = sizeof(outcfg);
        outcfg.format = AML_OUTDATA_FLOAT32;
        nn_output* out = (nn_output*)aml_module_output_get(ctx, outcfg);

        if (out && out->num > 0) {
            float* data = (float*)out->out[0].buf;
            int size = out->out[0].size / sizeof(float);
            postprocess_topk(data, size, labels, 5);
        }
        std::cout << "============================================================\n" << std::endl;
    }

    aml_module_destroy(ctx);
    return 0;
}
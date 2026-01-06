#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <float.h>
#include "nn_sdk.h"
#include "postprocess.h"

namespace fs = std::filesystem;

static void hwc_to_chw(const cv::Mat& src, float* dst) {
    int h = src.rows, w = src.cols;
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                dst[k * h * w + i * w + j] = src.at<cv::Vec3f>(i, j)[k];
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cout << "Usage: " << argv[0] << " <model.adla> <image_dir>\n"; return 0; }

    aml_config cfg{};
    cfg.typeSize = sizeof(cfg); cfg.modelType = ADLA_LOADABLE; cfg.nbgType = NN_ADLA_FILE; cfg.path = argv[1];
    void* ctx = aml_module_create(&cfg);
    if (!ctx) return -1;

    std::vector<float> chw_buffer(kInputW * kInputH * 3);
    fs::create_directory("yolo11_result");

    for (auto& it : fs::directory_iterator(argv[2])) {
        cv::Mat img = cv::imread(it.path().string());
        if (img.empty()) continue;

        std::cout << "============================================================" << std::endl;
        std::cout << "Processing image: \"" << it.path().filename().string() << "\"" << std::endl;
        std::cout << "============================================================" << std::endl;

        float scale = std::min((float)kInputW / img.cols, (float)kInputH / img.rows);
        int nw = img.cols * scale, nh = img.rows * scale;
        int px = (kInputW - nw) / 2, py = (kInputH - nh) / 2;
        cv::Mat res, canvas = cv::Mat::zeros(kInputH, kInputW, CV_32FC3);
        canvas.setTo(cv::Scalar(114.0/255.0, 114.0/255.0, 114.0/255.0)); 
        cv::resize(img, res, {nw, nh});
        res.convertTo(res, CV_32FC3, 1.0 / 255.0); 
        res.copyTo(canvas(cv::Rect(px, py, nw, nh)));
        hwc_to_chw(canvas, chw_buffer.data());

        nn_input in{}; in.typeSize = sizeof(in); in.input_type = BINARY_RAW_DATA;
        in.input = (unsigned char*)chw_buffer.data(); in.size = chw_buffer.size() * 4;
        in.info.valid = 1; in.info.input_format = AML_INPUT_MODEL_NCHW; in.info.input_data_type = AML_INPUT_FP32;
        aml_module_input_set(ctx, &in);

        aml_output_config_t outcfg{}; outcfg.typeSize = sizeof(outcfg); outcfg.format = AML_OUTDATA_FLOAT32;
        nn_output* out = (nn_output*)aml_module_output_get(ctx, outcfg);
        if (!out) continue;

        std::vector<cv::Rect> bboxes;
        std::vector<float> confs;
        std::vector<int> class_ids;
        std::vector<int> strides = {32, 16, 8};

        for (int i = 0; i < out->num; i++) {
            float* data = (float*)out->out[i].buf;
            int stride = strides[i], grid_h = kInputH / stride, grid_w = kInputW / stride;
            for (int g = 0; g < grid_h * grid_w; g++) {
                float* feat = data + g * kTotalChannels;
                float max_score = -1.0f; int cls_id = -1;
                for (int c = 0; c < kNumClasses; c++) {
                    float score = 1.0f / (1.0f + std::exp(-feat[64 + c]));
                    if (score > max_score) { max_score = score; cls_id = c; }
                }
                if (max_score > 0.3f) {
                    float d_l = decode_dfl(feat + 0), d_t = decode_dfl(feat + 16);
                    float d_r = decode_dfl(feat + 32), d_b = decode_dfl(feat + 48);
                    float cx = (g % grid_w) + 0.5f, cy = (g / grid_w) + 0.5f;
                    int rx1 = std::max(0, (int)(((cx - d_l) * stride - px) / scale));
                    int ry1 = std::max(0, (int)(((cy - d_t) * stride - py) / scale));
                    int rx2 = std::min(img.cols, (int)(((cx + d_r) * stride - px) / scale));
                    int ry2 = std::min(img.rows, (int)(((cy + d_b) * stride - py) / scale));
                    bboxes.push_back(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
                    confs.push_back(max_score); class_ids.push_back(cls_id);
                }
            }
        }

        std::vector<int> indices = manual_nms(bboxes, confs, 0.45f);
        if (!indices.empty()) {
            std::cout << "Detected " << indices.size() << " objects:" << std::endl;
            for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                printf("  %zu. %s (%.2f)\n", i + 1, kClassNames[class_ids[idx]].c_str(), confs[idx]);
                
                cv::rectangle(img, bboxes[idx], {0, 255, 0}, 2);
                char text[256]; std::sprintf(text, "%s %.2f", kClassNames[class_ids[idx]].c_str(), confs[idx]);
                cv::putText(img, text, {bboxes[idx].x, bboxes[idx].y - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
            }
        } else {
            std::cout << "No objects detected." << std::endl;
        }

        std::string out_path = "yolo11_result/" + it.path().filename().string();
        cv::imwrite(out_path, img);
        std::cout << "Result saved to: " << out_path << std::endl;
        std::cout << "============================================================" << std::endl << std::endl;
    }

    aml_module_destroy(ctx); return 0;
}
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
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <utility>
#include "model_loader.h"
#include "postprocess.h"

// TextDetector
TextDetector::TextDetector(const fs::path model_path) {
    model_path_ = model_path;
}

int TextDetector::InitNetwork() {
    std::cout << "TextDetector: Loading model from: " << model_path_ << std::endl;

    void* ctx_ = init_network(model_path_.c_str());
    if (!ctx_) {
        std::cerr << "TextDetector: Failed to initialize text detector with model: " << model_path_ << std::endl;
        throw std::runtime_error("Failed to initialize text detector");
    }
    return 0;
}

int TextDetector::UninitNetwork() {
   if (ctx_) {
        uninit_network(ctx_);
        std::cout << "TextDetector: Text detector uninitialized successfully." << std::endl;
        ctx_ = nullptr;
    }
    return 0;
}

cv::Mat TextDetector::Preprocess(const cv::Mat &image) {
    // the image is expected to be in BGR and NHWC format
    if (image.empty()) {
        std::cerr << "TextDetector: Input image is empty" << std::endl;
        return cv::Mat();
    }

    // resize the image to fit the model input size while keeping the aspect ratio, and pad to 960x960
    int target_size = 960;
    int src_w = image.cols;
    int src_h = image.rows;
    float resize_scale = std::min(static_cast<float>(target_size) / src_w, static_cast<float>(target_size) / src_h);
    resized_w_ = static_cast<int>(std::round(src_w * resize_scale));
    resized_h_ = static_cast<int>(std::round(src_h * resize_scale));
    resized_w_ = std::min(std::max(32, (resized_w_ + 31) / 32 * 32), target_size);
    resized_h_ = std::min(std::max(32, (resized_h_ + 31) / 32 * 32), target_size);
    resize_scale_x_ = (float)resized_w_ / src_w;
    resize_scale_y_ = (float)resized_h_ / src_h;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w_, resized_h_));
    cv::Mat padded = cv::Mat::zeros(target_size, target_size, image.type());
    resized.copyTo(padded(cv::Rect(0, 0, resized_w_, resized_h_)));

    // normalize
    cv::Mat normalized;
    padded.convertTo(normalized, CV_32F);
    cv::Scalar mean(123.675, 116.28, 103.53);
    cv::Scalar std(58.395, 57.12, 57.375);
    normalized = normalized - mean;
    cv::divide(normalized, std, normalized);

    float minf = 999.0f, maxf = -999.0f;
    for (int h = 0; h < normalized.rows; ++h) {
        for (int w = 0; w < normalized.cols; ++w) {
            cv::Vec3f pixel = normalized.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; ++c) {
                minf = std::min(minf, pixel[c]);
                maxf = std::max(maxf, pixel[c]);
            }
        }
    }
    std::cout << "TextDetector: normalized range is " << minf << " ~ " << maxf << std::endl;

    // quantize to int8
    float scale = 0.018723;
    float zero_point = -14.49078329023180;

    cv::Mat quantized(normalized.rows, normalized.cols, CV_8SC3);
    const float* src = (float*)normalized.data;
    int8_t* dst = (int8_t*)quantized.data;
    int total = normalized.total() * 3;

    for (int i = 0; i < total; ++i) {
        int q = static_cast<int>(std::round(src[i] / scale + zero_point));
        q = std::max(-128, std::min(127, q));
        dst[i] = static_cast<int8_t>(q);
    }

    int minv = 127, maxv = -128;
    for (int i = 0; i < total; ++i) {
        minv = std::min(minv, (int)dst[i]);
        maxv = std::max(maxv, (int)dst[i]);
    }

    // print the quantized values and their range
    std::cout << "TextDetector: quantized int8 range: " << minv << " ~ " << maxv << std::endl;
    return quantized;
}

std::pair<std::vector<cv::Point2f>, float> TextDetector::GetMiniBoxes(const std::vector<cv::Point>& contour) {
    cv::RotatedRect bounding_box = cv::minAreaRect(contour);
    cv::Point2f vertices[4];
    bounding_box.points(vertices);

    std::vector<std::pair<cv::Point2f, int>> points_with_idx;
    for (int i = 0; i < 4; ++i) {
        points_with_idx.push_back({vertices[i], i});
    }
    std::sort(points_with_idx.begin(), points_with_idx.end(),
              [](const auto& a, const auto& b) { return a.first.x < b.first.x; });

    int index_1, index_2, index_3, index_4;
    if (points_with_idx[1].first.y > points_with_idx[0].first.y) {
        index_1 = 0; index_4 = 1;
    } else {
        index_1 = 1; index_4 = 0;
    }
    if (points_with_idx[3].first.y > points_with_idx[2].first.y) {
        index_2 = 2; index_3 = 3;
    } else {
        index_2 = 3; index_3 = 2;
    }

    std::vector<cv::Point2f> box = {
        points_with_idx[index_1].first,
        points_with_idx[index_2].first,
        points_with_idx[index_3].first,
        points_with_idx[index_4].first
    };

    float min_side = std::min(bounding_box.size.width, bounding_box.size.height);
    return {box, min_side};
}

float TextDetector::BoxScoreFast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box) {
    int h = bitmap.rows;
    int w = bitmap.cols;

    float xmin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::min();
    float ymin = std::numeric_limits<float>::max();
    float ymax = std::numeric_limits<float>::min();

    for (const auto& point : box) {
        xmin = std::min(xmin, point.x);
        xmax = std::max(xmax, point.x);
        ymin = std::min(ymin, point.y);
        ymax = std::max(ymax, point.y);
    }

    int x_min = std::max(0, static_cast<int>(std::floor(xmin)));
    int x_max = std::min(w - 1, static_cast<int>(std::ceil(xmax)));
    int y_min = std::max(0, static_cast<int>(std::floor(ymin)));
    int y_max = std::min(h - 1, static_cast<int>(std::ceil(ymax)));

    if (x_max <= x_min || y_max <= y_min) return 0.0f;

    cv::Mat mask = cv::Mat::zeros(y_max - y_min + 1, x_max - x_min + 1, CV_8UC1);

    std::vector<cv::Point> contour;
    for (const auto& point : box) {
        contour.push_back(cv::Point(static_cast<int>(point.x - x_min),
                                   static_cast<int>(point.y - y_min)));
    }

    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{contour}, cv::Scalar(255));

    cv::Mat roi = bitmap(cv::Rect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1));
    cv::Scalar mean_val = cv::mean(roi, mask);
    return mean_val[0];
}

std::vector<cv::Point2f> TextDetector::UnclipPolygon(const std::vector<cv::Point2f>& box, float unclip_ratio) {

    std::vector<cv::Point> int_box;
    for (const auto& point : box) {
        int_box.push_back(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
    }

    double area = cv::contourArea(int_box);
    double perimeter = cv::arcLength(int_box, true);

    if (perimeter <= 0) return box;

    double distance = area * unclip_ratio / perimeter;


    cv::Point2f center(0, 0);
    for (const auto& point : box) {
        center.x += point.x;
        center.y += point.y;
    }
    center.x /= box.size();
    center.y /= box.size();

    std::vector<cv::Point2f> expanded_box;
    for (const auto& point : box) {
        cv::Point2f direction = point - center;
        float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
        if (length > 0) {
            direction.x /= length;
            direction.y /= length;
            cv::Point2f expanded_point = point + direction * static_cast<float>(distance);
            expanded_box.push_back(expanded_point);
        } else {
            expanded_box.push_back(point);
        }
    }

    return expanded_box;
}

std::vector<DetectionResult> TextDetector::Postprocess(const cv::Mat& image, const std::vector<float>& data) {
    // convert int8 data to float and reshape to 2D
    // the output is expected to be in NHWC format with shape [1, 1, height, width]
    cv::Mat pred_map(960, 960, CV_32F, (void*)data.data());

    // binarize the prediction map to get text regions
    cv::Mat bitmap;
    cv::threshold(pred_map, bitmap, 0.2, 1.0, cv::THRESH_BINARY);
    bitmap.convertTo(bitmap, CV_8UC1, 255.0);

    // find contours in the binary map
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    float box_thresh = 0.6;
    int max_candidates = 1000;
    float unclip_ratio = 1.5;
    int min_size = 3;
    int num_contours = std::min(static_cast<int>(contours.size()), max_candidates);
    int src_h = image.rows;
    int src_w = image.cols;
    std::vector<DetectionResult> results;

    // process each contour to get the final text boxes
    std::cout << "TextDetector: Processing " << num_contours << " contours..." << std::endl;
    for (int i = 0; i < num_contours; ++i) {
        const auto& contour = contours[i];

        // filter out small contours and low-score boxes
        auto [points, sside] = GetMiniBoxes(contour);
        if (sside < min_size) {
            continue;
        }
        float score = BoxScoreFast(pred_map, points);
        if (score < box_thresh) {
            continue;
        }

        // expand the box by unclip_ratio to get a better text region
        std::vector<cv::Point2f> expanded_points = UnclipPolygon(points, unclip_ratio);
        std::vector<cv::Point> expanded_contour;
        for (const auto& point : expanded_points) {
            expanded_contour.push_back(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
        }

        // get the final box and its score, and filter out small boxes
        auto [final_box, final_sside] = GetMiniBoxes(expanded_contour);
        if (final_sside < min_size + 2) {
            continue;
        }

        // scale the box back to the original image size
        std::vector<cv::Point2f> scaled_points;
        for (const auto& point : final_box) {
            cv::Point2f scaled_point;

            if (point.x < 0 || point.x >= resized_w_ ||
                point.y < 0 || point.y >= resized_h_) {
                continue;
            }

            scaled_point.x = point.x / resize_scale_x_;
            scaled_point.y = point.y / resize_scale_y_;

            scaled_point.x = std::max(0.0f, std::min((float)(src_w - 1), scaled_point.x));
            scaled_point.y = std::max(0.0f, std::min((float)(src_h - 1), scaled_point.y));

            scaled_points.push_back(scaled_point);
        }

        // further expand the box height to better cover the text region
        if (scaled_points.size() == 4) {
            cv::Point2f h_vec0 = scaled_points[3] - scaled_points[0];
            cv::Point2f h_vec1 = scaled_points[2] - scaled_points[1];
            cv::Point2f h_vec = (h_vec0 + h_vec1) * 0.5f;
            float h_len = std::sqrt(h_vec.x * h_vec.x + h_vec.y * h_vec.y);
            if (h_len > 1e-6f) {
                cv::Point2f h_unit(h_vec.x / h_len, h_vec.y / h_len);
                float delta = 0.3f * h_len;

                scaled_points[0].x -= h_unit.x * delta;
                scaled_points[0].y -= h_unit.y * delta;
                scaled_points[1].x -= h_unit.x * delta;
                scaled_points[1].y -= h_unit.y * delta;

                scaled_points[2].x += h_unit.x * delta;
                scaled_points[2].y += h_unit.y * delta;
                scaled_points[3].x += h_unit.x * delta;
                scaled_points[3].y += h_unit.y * delta;

                for (auto& p : scaled_points) {
                    p.x = std::max(0.0f, std::min(static_cast<float>(src_w - 1), p.x));
                    p.y = std::max(0.0f, std::min(static_cast<float>(src_h - 1), p.y));
                }
            }
        }

        DetectionResult result;
        for (const auto& p : scaled_points) {
            result.points.push_back(p);
        }

        // filter out boxes that are too small after scaling
        float rect_width = std::hypot(result.points[0].x - result.points[1].x,
                                      result.points[0].y - result.points[1].y);
        float rect_height = std::hypot(result.points[0].x - result.points[3].x,
                                       result.points[0].y - result.points[3].y);
        if (rect_width <= 3.0f || rect_height <= 3.0f) {
            continue;
        }
        result.score = score;

        results.push_back(result);
    }
    std::cout << "TextDetector: Finished processing contours. Detected " << results.size() << " text boxes." << std::endl;
    return results;
}

std::vector<DetectionResult> TextDetector::Detect(const cv::Mat &image) {
    cv::Mat preprocessed = Preprocess(image);
    if (preprocessed.rows != 960 ||
        preprocessed.cols != 960 ||
        preprocessed.channels() != 3 ||
        preprocessed.type() != CV_8SC3) {
        std::cerr << "TextDetector: invalid preprocessed shape: "
                  << preprocessed.cols << "x" << preprocessed.rows
                  << ", channels=" << preprocessed.channels()
                  << ", type=" << preprocessed.type()
                  << std::endl;
        return {};
    }

    nn_input inData;
    memset(&inData, 0, sizeof(nn_input));
    inData.input_type = BINARY_RAW_DATA;
    inData.input = preprocessed.data;
    inData.input_index = 0;
    inData.size = preprocessed.total() * preprocessed.elemSize();

    int ret = aml_module_input_set(ctx_, &inData);
    if (ret) {
        std::cerr << "Error: Failed to set input for text detection. Ret=" << ret << std::endl;
        return {};
    }

    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.format = AML_OUTDATA_FLOAT32;
    nn_output* nnout = (nn_output *)aml_module_output_get(ctx_, outconfig);
    if (!nnout) {
        std::cerr << "Error: Inference failed for text detection" << std::endl;
        return {};
    }

    float* out0 = reinterpret_cast<float*>(nnout->out[0].buf);
    size_t output_size = nnout->out[0].size / sizeof(float);
    std::vector<float> output_vector(out0, out0 + output_size);

    float min_val = FLT_MAX, max_val = -FLT_MAX, mean = 0;
    for (float v : output_vector) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        mean += v;
    }
    mean /= output_vector.size();
    std::cout << "TextDetector: aml module data stats: min=" << min_val
              << " max=" << max_val
              << " mean=" << mean << std::endl;
    return Postprocess(image, output_vector);
}

// TextRecognizer
bool TextRecognizer::LoadCharacterDict(const std::string& dict_path) {
    std::ifstream file(dict_path);
    if (!file.is_open()) {
        std::cerr << "TextRecognizer: Could not open dictionary file: " << dict_path << std::endl;
        return false;
    }

    character_dict_.clear();

    std::vector<std::string> character_str;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
        }
        if (!line.empty()) {
            character_str.push_back(line);
        }
    }

    bool use_space_char = true;
    if (use_space_char) {
        character_str.push_back(" ");
    }

    character_dict_.push_back("blank");
    for (const auto& char_str : character_str) {
        character_dict_.push_back(char_str);
    }

    return true;
}

TextRecognizer::TextRecognizer(const fs::path model_path, const fs::path dict_path) {
    model_path_ = model_path;
    dict_path_ = dict_path;
}

int TextRecognizer::InitNetwork() {
    std::cout << "TextRecognizer: Loading model from: " << model_path_ << std::endl;

    ctx_ = init_network(model_path_.c_str());
    if (!ctx_) {
        std::cerr << "TextRecognizer: Failed to initialize text recognizer with model: " << model_path_ << std::endl;
        throw std::runtime_error("Failed to initialize text recognizer");
    }

    if (!LoadCharacterDict(dict_path_.string())) {
        std::cerr << "TextRecognizer: Failed to load character dictionary for text recognition" << std::endl;
        uninit_network(ctx_);
        ctx_ = nullptr;
        throw std::runtime_error("Failed to load character dictionary");
    }
    return 0;
}

int TextRecognizer::UninitNetwork() {
    if (ctx_) {
        uninit_network(ctx_);
        std::cout << "TextRecognizer: Text recognizer uninitialized successfully." << std::endl;
        ctx_ = nullptr;
    }
    return 0;
}

cv::Mat TextRecognizer::Preprocess(const cv::Mat &image) {
    int img_h = 48, img_w = 320, img_c = 3;
    int h = image.rows;
    int w = image.cols;

    // risze
    float ratio = static_cast<float>(w) / static_cast<float>(h);
    int resized_w;
    if (std::ceil(img_h * ratio) > img_w) {
        resized_w = img_w;
    } else {
        resized_w = static_cast<int>(std::ceil(img_h * ratio));
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, img_h));

    // normalize
    resized.convertTo(resized, CV_32F);
    resized = resized / 255.0f;
    resized = (resized - 0.5f) / 0.5f;

    // pad to (48, 320, 3)
    cv::Mat output = cv::Mat::zeros(img_h, img_w, CV_32FC3);
    resized.copyTo(output(cv::Rect(0, 0, resized_w, img_h)));

    // quantize to int8
    float scale = 0.007843137254902;
    float zero_point = 0.4999999999993463;

    cv::Mat quantized(output.rows, output.cols, CV_8SC3);
    const float* src = (float*)output.data;
    int8_t* dst = (int8_t*)quantized.data;
    int total = output.total() * 3;

    for (int i = 0; i < total; ++i) {
        int q = static_cast<int>(std::round(src[i] / scale + zero_point));
        q = std::max(-128, std::min(127, q));
        dst[i] = static_cast<int8_t>(q);
    }

    double min_val, max_val;
    cv::minMaxLoc(quantized.reshape(1), &min_val, &max_val);
    std::cout << "TextRecognizer: quantized range: min=" << min_val << " max=" << max_val << std::endl;

    return quantized;
}

std::pair<std::string, float> TextRecognizer::Postprocess(const std::vector<float>& output_data) {
    int model_num_classes = static_cast<int>(character_dict_.size());
    if (model_num_classes <= 0) {
        std::cerr << "TextRecognizer: character_dict_ is empty" << std::endl;
        return {"", 0.0f};
    }

    if (output_data.size() % model_num_classes != 0) {
        std::cerr << "TextRecognizer: Output data size " << output_data.size()
                  << " is not divisible by num_classes " << model_num_classes << std::endl;
        return {"", 0.0f};
    }
    int seq_len = static_cast<int>(output_data.size() / model_num_classes);

    std::vector<int> preds_idx(seq_len);
    std::vector<float> preds_prob(seq_len);

    // for each time step, find the class with the highest probability
    for (int t = 0; t < seq_len; ++t) {
        const float* row = output_data.data() + t * model_num_classes;
        double sum_prob = 0.0;
        // compute the sum of probabilities for this time step
        for (int c = 0; c < model_num_classes; ++c) {
            sum_prob += row[c];
        }

        // apply softmax to get probabilities
        std::vector<float> probs(model_num_classes);
        if (sum_prob > 0.999 && sum_prob < 1.001) {
            // if the sum is close to 1, we assume the output is already probabilities and skip softmax
            for (int c = 0; c < model_num_classes; ++c) probs[c] = row[c];
        } else {
            // for numerical stability, we subtract the max logit before applying exp
            float max_logit = row[0];
            for (int c = 1; c < model_num_classes; ++c) max_logit = std::max(max_logit, row[c]);
            double exp_sum = 0.0;
            for (int c = 0; c < model_num_classes; ++c) {
                probs[c] = std::exp(row[c] - max_logit);
                exp_sum += probs[c];
            }
            float inv_sum = exp_sum > 0 ? static_cast<float>(1.0 / exp_sum) : 0.0f;
            for (int c = 0; c < model_num_classes; ++c) probs[c] *= inv_sum;
        }

        // find the class with the highest probability
        int best_idx = 0;
        float best_prob = probs[0];
        for (int c = 1; c < model_num_classes; ++c) {
            if (probs[c] > best_prob) {
                best_prob = probs[c];
                best_idx = c;
            }
        }
        preds_idx[t] = best_idx;
        preds_prob[t] = best_prob;
    }

    std::string text;
    std::vector<float> conf_list;
    std::vector<bool> selection(seq_len, true);

    // remove duplicates
    for (int i = 1; i < seq_len; ++i) {
        selection[i] = preds_idx[i] != preds_idx[i - 1];
    }

    // remove blank tokens (index 0 is reserved for blank)
    for (int i = 0; i < seq_len; ++i) {
        selection[i] = selection[i] && (preds_idx[i] != 0);
    }

    // construct the final text and confidence list based on the selected indices
    std::ostringstream dbg_chars;
    bool dbg_started = false;
    for (int i = 0; i < seq_len; ++i) {
        if (selection[i]) {
            int idx = preds_idx[i];

            if (idx > 0 && idx < static_cast<int>(character_dict_.size())) {
                const std::string& ch = character_dict_[idx];
                text += ch;
                conf_list.push_back(preds_prob[i]);
                if (!dbg_started) { dbg_started = true; dbg_chars << "Selected chars: "; }
                dbg_chars << "[i=" << i << ", idx=" << idx << ", ch='" << ch << "', p="
                          << std::fixed << std::setprecision(4) << preds_prob[i] << "] ";
            }
        }
    }

    // compute average confidence score for the recognized text
    float avg_confidence = 0.0f;
    if (!conf_list.empty()) {
        float sum = 0.0f;
        for (float conf : conf_list) {
            sum += conf;
        }
        avg_confidence = sum / conf_list.size();
    } else {
        avg_confidence = 0.0f;
    }

    std::cout << "TextRecognizer: Postprocessing completed. Recognized text: '" << text
              << "', average confidence: " << std::fixed << std::setprecision(4) << avg_confidence
              << std::endl;
    return {text, avg_confidence};
}

std::string TextRecognizer::Recognize(const cv::Mat &cropped_image) {
    cv::Mat preprocessed = Preprocess(cropped_image);
    if (preprocessed.rows != 48 ||
        preprocessed.cols != 320 ||
        preprocessed.channels() != 3 ||
        preprocessed.type() != CV_8SC3) {

        std::cerr << "Recognition: invalid preprocessed shape: "
                << preprocessed.cols << "x" << preprocessed.rows
                << ", channels=" << preprocessed.channels()
                << ", type=" << preprocessed.type()
                << std::endl;

        return "";
    }

    nn_input inData;
    memset(&inData, 0, sizeof(nn_input));
    inData.input_type = BINARY_RAW_DATA;
    inData.input = preprocessed.data;
    inData.input_index = 0;
    inData.size = preprocessed.cols * preprocessed.rows * preprocessed.channels() * sizeof(uint8_t);

    int ret = aml_module_input_set(ctx_, &inData);

    aml_output_config_t outconfig;
    memset(&outconfig, 0, sizeof(aml_output_config_t));
    outconfig.typeSize = sizeof(aml_output_config_t);
    outconfig.format = AML_OUTDATA_FLOAT32;
    nn_output* nnout = (nn_output *)aml_module_output_get(ctx_, outconfig);
    if (!nnout) {
        std::cerr << "Error: Inference failed for text recognition" << std::endl;
        return "";
    }

    float* output_data = reinterpret_cast<float*>(nnout->out[0].buf);
    size_t output_size = nnout->out[0].size / sizeof(float);
    std::vector<float> output_vec(output_data, output_data + output_size);
    auto result = Postprocess(output_vec);
    return result.first;
}

// OcrEngine
OcrEngine::OcrEngine(const fs::path& det_model_path, const fs::path& rec_model_path, const fs::path& dict_path)
    : detector_(det_model_path), recognizer_(rec_model_path, dict_path) {
}

std::vector<DetectionResult> OcrEngine::SortBoxes(const std::vector<DetectionResult>& boxes) {
    std::vector<DetectionResult> sorted_boxes = boxes;

    std::sort(sorted_boxes.begin(), sorted_boxes.end(),
              [](const DetectionResult& a, const DetectionResult& b) {
                  float y_a = (a.points[0].y + a.points[2].y) / 2;
                  float y_b = (b.points[0].y + b.points[2].y) / 2;

                  if (std::abs(y_a - y_b) < 10) {
                      float x_a = (a.points[0].x + a.points[2].x) / 2;
                      float x_b = (b.points[0].x + b.points[2].x) / 2;
                      return x_a < x_b;
                  }
                  return y_a < y_b;
              });

    return sorted_boxes;
}

cv::Mat OcrEngine::GetRotateCropImage(const cv::Mat& image, const DetectionResult& box) {
    std::vector<cv::Point2f> points = box.points;
    if (points.size() != 4) {
        std::cerr << "OcrEngine: TextBox must have exactly 4 points, got " << points.size() << std::endl;
        return cv::Mat();
    }

    int img_crop_width = static_cast<int>(std::max(
        cv::norm(points[0] - points[1]),
        cv::norm(points[2] - points[3])
    ));
    int img_crop_height = static_cast<int>(std::max(
        cv::norm(points[0] - points[3]),
        cv::norm(points[1] - points[2])
    ));

    std::vector<cv::Point2f> pts_std = {
        cv::Point2f(0, 0),
        cv::Point2f(img_crop_width, 0),
        cv::Point2f(img_crop_width, img_crop_height),
        cv::Point2f(0, img_crop_height)
    };

    cv::Mat M = cv::getPerspectiveTransform(points, pts_std);
    cv::Mat dst_img;
    cv::warpPerspective(image, dst_img, M, cv::Size(img_crop_width, img_crop_height),
                       cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    int dst_img_height = dst_img.rows;
    int dst_img_width = dst_img.cols;
    if (dst_img_height * 1.0 / dst_img_width >= 1.5) {
        cv::Mat rotated;
        cv::rotate(dst_img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        return rotated;
    }
    return dst_img;
}

std::vector<OCRResult> OcrEngine::Process(const cv::Mat& image) {
    detector_.InitNetwork();
    auto det_results = detector_.Detect(image);
    if (det_results.empty()) {
        std::cout << "OcrEngine: No text detected in the image." << std::endl;
        return {};
    }
    detector_.UninitNetwork();
    det_results = SortBoxes(det_results);

    std::vector<OCRResult> ocr_results;
    recognizer_.InitNetwork();
    for (const auto& det_result : det_results) {
        cv::Mat cropped_img = GetRotateCropImage(image, det_result);
        if (cropped_img.empty()) continue;

        std::string text = recognizer_.Recognize(cropped_img);

        OCRResult result;
        result.box = det_result;
        result.text = text;
        result.score = det_result.score;
        ocr_results.push_back(result);
    }
    recognizer_.UninitNetwork();

    return ocr_results;
}

// OcrUtils
int OcrUtils::DrawBoxes(cv::Mat& image, const std::vector<DetectionResult> &boxes, float drop_score) {
    for (const auto& box : boxes) {
        if (box.score < drop_score) continue;

        const auto& pts = box.points;
        for (int i = 0; i < 4; ++i) {
            cv::line(image, pts[i], pts[(i + 1) % 4],
                     cv::Scalar(0, 255, 0), 2);
        }
    }

    return 0;
}

int OcrUtils::DrawOCRResults(cv::Mat& image, const std::vector<OCRResult> &results, float drop_score) {


    for (const auto& res : results) {
        if (res.score < drop_score) continue;

        const auto& pts = res.box.points;
        for (int i = 0; i < 4; ++i) {
            cv::line(image, pts[i], pts[(i + 1) % 4],
                     cv::Scalar(0, 255, 0), 2);
        }

        float min_x = image.cols, min_y = image.rows;
        for (const auto& p : pts) {
            min_x = std::min(min_x, p.x);
            min_y = std::min(min_y, p.y);
        }

        int x = (int)min_x;
        int y = (int)min_y - 5;
        if (y < 10) y = min_y + 15;

        cv::putText(image, res.text,
                    cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 0, 0),
                    2);

        std::cout << "OCR: " << res.text
                  << " (" << res.score << ")" << std::endl;
    }

    return 0;
}
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

#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <numeric>

std::tuple<cv::Mat, float, std::tuple<int, int>> preproc(const cv::Mat& img, std::tuple<int, int> input_size) {
    // 1. letterbox (resize + padding)
    // 2. BGR to RGB conversion
    // 3. Normalize to 0-1 (divide by 255.0)
    // Note: NNSDK's model_loader expects HWC format, so return HWC instead of CHW
    
    int input_height = std::get<0>(input_size);
    int input_width = std::get<1>(input_size);

    // letterbox: calculate scale and padding
    float scale = std::min(static_cast<float>(input_height) / img.rows,
                           static_cast<float>(input_width) / img.cols);
    int new_w = static_cast<int>(std::round(img.cols * scale));
    int new_h = static_cast<int>(std::round(img.rows * scale));
    
    // resize
    cv::Mat resized_img;
    if (img.size() != cv::Size(new_w, new_h)) {
        cv::resize(img, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized_img = img.clone();
    }
    
    // padding
    float pad_w = (input_width - new_w) / 2.0f;
    float pad_h = (input_height - new_h) / 2.0f;
    int top = static_cast<int>(std::round(pad_h - 0.1f));
    int bottom = static_cast<int>(std::round(pad_h + 0.1f));
    int left = static_cast<int>(std::round(pad_w - 0.1f));
    int right = static_cast<int>(std::round(pad_w + 0.1f));
    
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // BGR to RGB conversion
    cv::Mat rgb_img;
    cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);
    
    // Normalize to 0-1 range (divide by 255.0)
    cv::Mat normalized_img;
    rgb_img.convertTo(normalized_img, CV_32F, 1.0 / 255.0);

    // mean = [0.485, 0.456, 0.406]
    // std = [0.229, 0.224, 0.225]
    // Note: Use cv::divide for per-channel division in OpenCV
    cv::Scalar mean(0.485f, 0.456f, 0.406f);
    cv::Scalar std(0.229f, 0.224f, 0.225f);
    normalized_img -= mean;
    cv::divide(normalized_img, std, normalized_img);
    
    // Return HWC format, ImageNet normalized float32 image (RGB format)
    // Also return scale and padding (left, top) for coordinate mapping
    return std::make_tuple(normalized_img, scale, std::make_tuple(left, top));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void demo_postprocess(float* outputs, int num_boxes, std::tuple<int, int> img_size, bool p6) {
    int img_height = std::get<0>(img_size);
    int img_width = std::get<1>(img_size);
    
    std::vector<int> strides;
    if (!p6) {
        strides = {8, 16, 32};
    } else {
        strides = {8, 16, 32, 64};
    }
    
    // Calculate grid count for each stride
    std::vector<int> hsizes, wsizes;
    for (int stride : strides) {
        hsizes.push_back(img_height / stride);
        wsizes.push_back(img_width / stride);
    }
    
    // Build grids and expanded_strides
    std::vector<std::vector<float>> grids_list;
    std::vector<std::vector<float>> strides_list;
    
    int total_boxes = 0;
    for (size_t i = 0; i < strides.size(); ++i) {
        int hsize = hsizes[i];
        int wsize = wsizes[i];
        int stride = strides[i];
        int grid_size = hsize * wsize;
        
        std::vector<float> grid(grid_size * 2);
        std::vector<float> expanded_stride(grid_size);
        
        for (int h = 0; h < hsize; ++h) {
            for (int w = 0; w < wsize; ++w) {
                int idx = h * wsize + w;
                grid[idx * 2] = static_cast<float>(w);
                grid[idx * 2 + 1] = static_cast<float>(h);
                expanded_stride[idx] = static_cast<float>(stride);
            }
        }
        
        grids_list.push_back(grid);
        strides_list.push_back(expanded_stride);
        total_boxes += grid_size;
    }
    
    // Merge all grids and strides
    std::vector<float> all_grids(total_boxes * 2);
    std::vector<float> all_strides(total_boxes);
    
    int offset = 0;
    for (size_t i = 0; i < grids_list.size(); ++i) {
        int grid_size = grids_list[i].size() / 2;
        for (int j = 0; j < grid_size; ++j) {
            all_grids[(offset + j) * 2] = grids_list[i][j * 2];
            all_grids[(offset + j) * 2 + 1] = grids_list[i][j * 2 + 1];
            all_strides[offset + j] = strides_list[i][j];
        }
        offset += grid_size;
    }
    
    // Apply grid and stride decoding
    for (int i = 0; i < num_boxes && i < total_boxes; ++i) {
        float* box = outputs + i * 85;
        
        // outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        box[0] = (box[0] + all_grids[i * 2]) * all_strides[i];
        box[1] = (box[1] + all_grids[i * 2 + 1]) * all_strides[i];
        
        // outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        box[2] = std::exp(box[2]) * all_strides[i];
        box[3] = std::exp(box[3]) * all_strides[i];
    }
}

std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float nms_thr) {
    if (boxes.empty()) return {};
    
    // Create indices and sort
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        float x1_i = boxes[idx].x;
        float y1_i = boxes[idx].y;
        float x2_i = boxes[idx].x + boxes[idx].width;
        float y2_i = boxes[idx].y + boxes[idx].height;
        float area_i = boxes[idx].width * boxes[idx].height;
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;
            
            float x1_j = boxes[idx_j].x;
            float y1_j = boxes[idx_j].y;
            float x2_j = boxes[idx_j].x + boxes[idx_j].width;
            float y2_j = boxes[idx_j].y + boxes[idx_j].height;
            
            float xx1 = std::max(x1_i, x1_j);
            float yy1 = std::max(y1_i, y1_j);
            float xx2 = std::min(x2_i, x2_j);
            float yy2 = std::min(y2_i, y2_j);
            
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;
            
            float area_j = boxes[idx_j].width * boxes[idx_j].height;
            float ovr = inter / (area_i + area_j - inter);
            
            if (ovr > nms_thr) {
                suppressed[idx_j] = true;
            }
        }
    }
    
    return keep;
}

std::vector<Detection> multiclass_nms(const std::vector<cv::Rect2f>& boxes, 
                                      const std::vector<std::vector<float>>& scores,
                                      int num_classes,
                                      float nms_thr, 
                                      float score_thr) {
    if (boxes.empty() || scores.empty()) return {};
    
    // Find max class score and class ID for each box
    std::vector<float> cls_scores(boxes.size());
    std::vector<int> cls_inds(boxes.size());
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        float max_score = -1.0f;
        int max_idx = -1;
        for (int c = 0; c < num_classes; ++c) {
            if (scores[i][c] > max_score) {
                max_score = scores[i][c];
                max_idx = c;
            }
        }
        cls_scores[i] = max_score;
        cls_inds[i] = max_idx;
    }
    
    // Filter low-score boxes
    std::vector<cv::Rect2f> valid_boxes;
    std::vector<float> valid_scores;
    std::vector<int> valid_cls_inds;
    std::vector<int> valid_indices;
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (cls_scores[i] > score_thr) {
            valid_boxes.push_back(boxes[i]);
            valid_scores.push_back(cls_scores[i]);
            valid_cls_inds.push_back(cls_inds[i]);
            valid_indices.push_back(i);
        }
    }
    
    if (valid_boxes.empty()) return {};
    
    // Execute NMS
    std::vector<int> keep = nms(valid_boxes, valid_scores, nms_thr);
    
    // Build results
    std::vector<Detection> dets;
    for (int idx : keep) {
        Detection det;
        det.x1 = valid_boxes[idx].x;
        det.y1 = valid_boxes[idx].y;
        det.x2 = valid_boxes[idx].x + valid_boxes[idx].width;
        det.y2 = valid_boxes[idx].y + valid_boxes[idx].height;
        det.score = valid_scores[idx];
        det.class_id = valid_cls_inds[idx];
        dets.push_back(det);
    }
    
    return dets;
}

cv::Mat vis(const cv::Mat& img, 
            const std::vector<Detection>& detections,
            float conf_thresh,
            const std::vector<std::string>& class_names) {
    cv::Mat result = img.clone();
    
    // Adjust font size based on image size
    int img_height = img.rows;
    int img_width = img.cols;
    float font_scale = std::max(0.6f, std::min(1.2f, 
        static_cast<float>(std::sqrt(img_height * img_height + img_width * img_width)) * 0.0015f));
    int thickness = std::max(2, static_cast<int>(font_scale * 2.5f));
    
    // YOLOX color palette
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 114, 189), cv::Scalar(217, 83, 25), cv::Scalar(237, 177, 32),
        cv::Scalar(126, 47, 142), cv::Scalar(119, 172, 48), cv::Scalar(77, 190, 238),
        cv::Scalar(162, 20, 47), cv::Scalar(77, 77, 77), cv::Scalar(153, 153, 153),
        cv::Scalar(255, 0, 0), cv::Scalar(255, 128, 0), cv::Scalar(191, 191, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(85, 85, 0), cv::Scalar(85, 170, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(170, 85, 0), cv::Scalar(170, 170, 0), cv::Scalar(170, 255, 0),
        cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0), cv::Scalar(255, 255, 0),
        cv::Scalar(0, 85, 128), cv::Scalar(0, 170, 128), cv::Scalar(0, 255, 128),
        cv::Scalar(85, 0, 128), cv::Scalar(85, 85, 128), cv::Scalar(85, 170, 128),
        cv::Scalar(85, 255, 128), cv::Scalar(170, 0, 128), cv::Scalar(170, 85, 128),
        cv::Scalar(170, 170, 128), cv::Scalar(170, 255, 128), cv::Scalar(255, 0, 128),
        cv::Scalar(255, 85, 128), cv::Scalar(255, 170, 128), cv::Scalar(255, 255, 128),
        cv::Scalar(0, 85, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 255, 255),
        cv::Scalar(85, 0, 255), cv::Scalar(85, 85, 255), cv::Scalar(85, 170, 255),
        cv::Scalar(85, 255, 255), cv::Scalar(170, 0, 255), cv::Scalar(170, 85, 255),
        cv::Scalar(170, 170, 255), cv::Scalar(170, 255, 255), cv::Scalar(255, 0, 255),
        cv::Scalar(255, 85, 255), cv::Scalar(255, 170, 255), cv::Scalar(85, 0, 0),
        cv::Scalar(128, 0, 0), cv::Scalar(170, 0, 0), cv::Scalar(213, 0, 0),
        cv::Scalar(255, 0, 0), cv::Scalar(0, 43, 0), cv::Scalar(0, 85, 0),
        cv::Scalar(0, 128, 0), cv::Scalar(0, 170, 0), cv::Scalar(0, 213, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 43), cv::Scalar(0, 0, 85),
        cv::Scalar(0, 0, 128), cv::Scalar(0, 0, 170), cv::Scalar(0, 0, 213),
        cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 0), cv::Scalar(36, 36, 36),
        cv::Scalar(219, 219, 219), cv::Scalar(255, 255, 255)
    };

    for (const auto& det : detections) {
        if (det.score < conf_thresh) continue;
        if (det.class_id < 0 || det.class_id >= static_cast<int>(class_names.size())) continue;
        
        int x0 = static_cast<int>(det.x1);
        int y0 = static_cast<int>(det.y1);
        int x1 = static_cast<int>(det.x2);
        int y1 = static_cast<int>(det.y2);
        
        cv::Scalar color = colors[det.class_id % colors.size()];
        
        // Draw bounding box
        cv::rectangle(result, cv::Point(x0, y0), cv::Point(x1, y1), color, thickness);
        
        // Prepare text
        std::string text = class_names[det.class_id] + ":" + cv::format("%.1f%%", det.score * 100);
        
        // Calculate text size
        int baseline = 0;
        cv::Size txt_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
        
        // Draw text background
        cv::Scalar txt_bk_color = color * 0.7;
        cv::rectangle(result,
                     cv::Point(x0, y0 + 1),
                     cv::Point(x0 + txt_size.width + 1, y0 + static_cast<int>(1.5 * txt_size.height)),
                     txt_bk_color, -1);
        
        // Draw text
        cv::Scalar txt_color = (cv::mean(color)[0] > 0.5) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
        cv::putText(result, text,
                   cv::Point(x0, y0 + txt_size.height),
                   cv::FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness);
    }
    
    return result;
}

void extract_boxes_and_scores(
    float* output,
    int num_boxes,
    int num_classes,
    float scale,
    int pad_left,
    int pad_top,
    int img_width,
    int img_height,
    std::vector<cv::Rect2f>& boxes,
    std::vector<std::vector<float>>& scores)
{
    boxes.clear();
    scores.clear();
    boxes.reserve(num_boxes);
    scores.reserve(num_boxes);

    // Extract all boxes and scores
    for (int i = 0; i < num_boxes; ++i) {
        float* box_data = output + i * 85;

        // Format after demo_postprocess: [cx, cy, w, h, obj_conf, class0, ..., class79]
        float cx = box_data[0];
        float cy = box_data[1];
        float w = box_data[2];
        float h = box_data[3];

        // Python: boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Python: valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
        // Python: valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale
        x1 = (x1 - pad_left) / scale;
        y1 = (y1 - pad_top) / scale;
        x2 = (x2 - pad_left) / scale;
        y2 = (y2 - pad_top) / scale;

        // Ensure coordinates are within image bounds
        x1 = std::max(0.0f, std::min(static_cast<float>(img_width), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(img_height), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(img_width), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(img_height), y2));

        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));

        // Calculate class scores (obj_conf * cls_scores)
        float obj_conf = box_data[4];
        std::vector<float> cls_scores(num_classes);
        for (int c = 0; c < num_classes; ++c) {
            float cls_score = box_data[5 + c];
            cls_scores[c] = obj_conf * cls_score;
        }
        scores.push_back(cls_scores);
    }
}
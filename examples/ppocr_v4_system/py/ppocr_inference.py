# -*- coding: utf-8 -*-

"""
Copyright (C) 2024-2025 Amlogic, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import os
import glob
import argparse
import cv2
import pyclipper
from pathlib import Path
from amlnn.api import AMLNN

# ==========================================
# CONFIGURATION
# ==========================================
DET_MODEL_WIDTH = 640
DET_MODEL_HEIGHT = 640
REC_MODEL_WIDTH = 320
REC_MODEL_HEIGHT = 48

BOX_THRESH = 0.3
BOX_SCORE_THRESH = 0.6
UNCLIP_RATIO = 1.5
MIN_SIZE = 3

DET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DET_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
REC_MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
REC_STD  = np.array([127.5, 127.5, 127.5], dtype=np.float32)


# ==========================================
# UTILS
# ==========================================
def load_dictionary(dict_path):
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
    dictionary = ['blank']
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            dictionary.append(line.strip('\r\n'))
    dictionary.append(' ')
    return dictionary

def cpp_round(x):
    """ Matches C++ std::round behavior (Round half away from zero) """
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5))

def apply_quantization(tensor, s, zp, tensor_type):
    """ Quantize float tensor to integer based on hardware attributes """
    if tensor_type == 0:
        return tensor # FP16

    val = cpp_round(tensor / s + zp)
    if tensor_type == 2:   # INT8
        return np.clip(val, -128, 127).astype(np.int8)
    elif tensor_type == 3: # UINT8
        return np.clip(val, 0, 255).astype(np.uint8)
    elif tensor_type == 4: # INT16
        return np.clip(val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}.")

def get_rotate_crop_image(img, points):
    """ Get the perspective transformed cropped image for slanted text """
    points = np.array(points, dtype=np.float32)
    img_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])))
    
    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height]
    ])
    
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    
    if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
        dst_img = np.rot90(dst_img)
        
    return dst_img

# ==========================================
# DET PIPELINE
# ==========================================
def preprocess_det(image, s, zp, tensor_type):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_img.shape[:2]

    # Max Ratio scaling
    ratio_max = max(orig_w / float(DET_MODEL_WIDTH), orig_h / float(DET_MODEL_HEIGHT))
    new_w = min(int(orig_w / ratio_max), DET_MODEL_WIDTH)
    new_h = min(int(orig_h / ratio_max), DET_MODEL_HEIGHT)

    resized_img = cv2.resize(rgb_img, (new_w, new_h))
    
    # Pad right and bottom
    padded_img = np.zeros((DET_MODEL_HEIGHT, DET_MODEL_WIDTH, 3), dtype=np.uint8)
    padded_img[0:new_h, 0:new_w] = resized_img

    # Normalize (ImageNet)
    float_img = padded_img.astype(np.float32)
    float_img = (float_img - DET_MEAN) / DET_STD

    # Format NHWC and Quantize
    input_tensor = np.expand_dims(float_img, axis=0)
    input_tensor = apply_quantization(input_tensor, s, zp, tensor_type)

    return input_tensor, ratio_max

def order_points_cpp_style(pts):
    """ Sorts points exactly how C++ cv_point_compare + index logic does """
    pts = pts[np.argsort(pts[:, 0])] # Sort by X
    if pts[1][1] > pts[0][1]:
        idx1, idx4 = 0, 1
    else:
        idx1, idx4 = 1, 0

    if pts[3][1] > pts[2][1]:
        idx2, idx3 = 2, 3
    else:
        idx2, idx3 = 3, 2

    return np.array([pts[idx1], pts[idx2], pts[idx3], pts[idx4]], dtype=np.float32)

def get_box_score_fast(pred_map, box):
    h, w = pred_map.shape
    box = np.array(box).reshape(-1, 2)
    min_x = np.clip(np.min(box[:, 0]), 0, w - 1)
    max_x = np.clip(np.max(box[:, 0]), 0, w - 1)
    min_y = np.clip(np.min(box[:, 1]), 0, h - 1)
    max_y = np.clip(np.max(box[:, 1]), 0, h - 1)
    
    mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    shifted_box = (box - [min_x, min_y]).astype(np.int32)
    cv2.fillPoly(mask, [shifted_box], 1)
    
    crop = pred_map[min_y:max_y+1, min_x:max_x+1]
    return cv2.mean(crop, mask=mask)[0]

def unclip(box, unclip_ratio):
    poly = box.astype(np.int32).tolist()
    distance = cv2.contourArea(box) * unclip_ratio / cv2.arcLength(box, True)
    
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    
    if len(expanded) == 0:
        return []
    return np.array(expanded[0], dtype=np.float32)

def postprocess_det(outputs, orig_shape, ratio_max):
    pred_map = np.squeeze(outputs[0])
    orig_h, orig_w = orig_shape
    
    # Binarization & Dilation
    bit_map = (pred_map > BOX_THRESH).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bit_map = cv2.dilate(bit_map, kernel, iterations=1)

    contours, _ = cv2.findContours(bit_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = min(len(contours), 1000) # MAX_CANDIDATES

    res_boxes = []
    for i in range(num_contours):
        contour = contours[i].squeeze(1) if len(contours[i].shape) > 2 else contours[i]
        if len(contour) <= 2:
            continue
            
        rect = cv2.minAreaRect(contour)
        min_side_len = min(rect[1])
        if min_side_len < MIN_SIZE:
            continue

        score = get_box_score_fast(pred_map, contour)
        if score < BOX_SCORE_THRESH:
            continue

        box_points = cv2.boxPoints(rect)
        box_points = order_points_cpp_style(box_points)

        clip_box = unclip(box_points, UNCLIP_RATIO)
        if len(clip_box) == 0:
            continue

        clip_rect = cv2.minAreaRect(clip_box)
        clip_min_side_len = min(clip_rect[1])
        if clip_min_side_len < MIN_SIZE + 2:
            continue

        clip_box_points = cv2.boxPoints(clip_rect)
        clip_box_points = order_points_cpp_style(clip_box_points)

        # Scale coordinates back to original image
        final_box = []
        for p in clip_box_points:
            x = min(max(int(p[0] * ratio_max), 0), orig_w)
            y = min(max(int(p[1] * ratio_max), 0), orig_h)
            final_box.append([x, y])
            
        res_boxes.append({'box': final_box})
        
    return res_boxes

# ==========================================
# REC PIPELINE
# ==========================================
def preprocess_rec(image, s, zp, tensor_type):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb_img.shape[:2]

    ratio = orig_w / float(orig_h)
    new_w = min(int(REC_MODEL_HEIGHT * ratio), REC_MODEL_WIDTH)
    new_w = max(1, new_w) 

    resized_img = cv2.resize(rgb_img, (new_w, REC_MODEL_HEIGHT))

    # Fix: Pad the right side with 0 (Black) BEFORE normalization, matching the working REC standalone code
    padded_img = np.zeros((REC_MODEL_HEIGHT, REC_MODEL_WIDTH, 3), dtype=np.uint8)
    padded_img[0:REC_MODEL_HEIGHT, 0:new_w] = resized_img

    # Normalize [-1.0, 1.0]
    float_img = padded_img.astype(np.float32)
    float_img = (float_img - REC_MEAN) / REC_STD

    # Format NHWC and Quantize
    input_tensor = np.expand_dims(float_img, axis=0)
    input_tensor = apply_quantization(input_tensor, s, zp, tensor_type)

    return input_tensor

def postprocess_rec(outputs, dictionary):
    pred_map = np.squeeze(outputs[0])
    if len(pred_map.shape) < 2: return "", 0.0

    seq_len = pred_map.shape[0]
    text = ""
    total_score = 0.0
    valid_count = 0
    pre_idx = -1

    for i in range(seq_len):
        max_idx = int(np.argmax(pred_map[i]))
        max_score = float(pred_map[i][max_idx])
        
        # CTC Rules: Ignore blank (index 0) and consecutive duplicates
        if max_idx > 0 and max_idx != pre_idx:
            # FIX: Removed the `char_idx = max_idx - 1` that misaligned the dictionary indices
            if max_idx < len(dictionary):
                text += dictionary[max_idx]
            total_score += max_score
            valid_count += 1
            
        pre_idx = max_idx

    avg_score = total_score / valid_count if valid_count > 0 else 0.0
    return text, avg_score

# ==========================================
# VISUALIZATION
# ==========================================
def draw_ocr_results(image, det_results):
    for obj in det_results:
        box = np.array(obj['box']).astype(np.int32)
        cv2.polylines(image, [box], True, (0, 255, 0), 2)
        
        text = obj.get('text', '')
        if text:
            # Added CTC score parsing cleanly
            score = obj.get('rec_score', 0.0)
            label = f"{text} ({score:.2f})"
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            text_size, baseline = cv2.getTextSize(label, font_face, font_scale, thickness)
            
            text_x, text_y = box[0][0], max(text_size[1], box[0][1] - 5)
            
            bg_top_left = (text_x, text_y - text_size[1] - 2)
            bg_bottom_right = (text_x + text_size[0], text_y + baseline + 2)
            
            # Draw green background and black text
            cv2.rectangle(image, bg_top_left, bg_bottom_right, (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (text_x, text_y), font_face, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
    return image

def main():
    parser = argparse.ArgumentParser(description="End-to-End PPOCR Demo")
    parser.add_argument('--det', required=True, help='Path to det .adla model')
    parser.add_argument('--rec', required=True, help='Path to rec .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--dict', required=True, help='Path to PP-OCR dictionary file')
    args = parser.parse_args()

    print("PPOCR End-to-End Demo")

    # 1. Load Dictionary
    char_dict = load_dictionary(args.dict)

    # 2. Initialize DET Network
    det_amlnn = AMLNN()
    det_amlnn.init_runtime(mode="native", enable_perf=True)
    det_amlnn.load_model(path=args.det)
    det_tensor_info = det_amlnn.get_tensor_info()
    print(f"DET SDK Version: {det_amlnn.get_sdk_version()}")
    
    det_attr = det_tensor_info["inputs"][0]
    det_s = float(det_attr["scale"])
    det_zp = int(det_attr["zp"])
    det_type = int(det_attr["type"])

    # 3. Initialize REC Network
    rec_amlnn = AMLNN()
    rec_amlnn.init_runtime(mode="native", enable_perf=True)
    rec_amlnn.load_model(path=args.rec)
    rec_tensor_info = rec_amlnn.get_tensor_info()
    print(f"REC SDK Version: {rec_amlnn.get_sdk_version()}")

    rec_attr = rec_tensor_info["inputs"][0]
    rec_s = float(rec_attr["scale"])
    rec_zp = int(rec_attr["zp"])
    rec_type = int(rec_attr["type"])

    # 4. Find Images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files.sort()

    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        det_amlnn.uninit()
        rec_amlnn.uninit()
        return

    # 5. Generate Output Directory
    result_dir = "ppocr_end2end_result"
    os.makedirs(result_dir, exist_ok=True)

    # 6. Process loop
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"can't read image: {image_path}")

            orig_h, orig_w = img.shape[:2]

            # --- DET Inference ---
            det_input, ratio_max = preprocess_det(img, det_s, det_zp, det_type)
            det_outputs = det_amlnn.inference(inputs=[det_input])
            det_results = postprocess_det(det_outputs, (orig_h, orig_w), ratio_max)

            print(f"    [RESULT] Detected {len(det_results)} objects.")
            print("    " + "-" * 56)

            # --- REC Inference ---
            for idx, obj in enumerate(det_results):
                box = obj['box']
                crop_img = get_rotate_crop_image(img, box)

                if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                    continue

                # Inference
                rec_input = preprocess_rec(crop_img, rec_s, rec_zp, rec_type)
                rec_outputs = rec_amlnn.inference(inputs=[rec_input])
                text, rec_score = postprocess_rec(rec_outputs, char_dict)

                obj['text'] = text
                obj['rec_score'] = rec_score
                print(f"    [RESULT] Box {idx} - Text: [{text}] (Score: {rec_score:.2f})")

            # --- Draw and Save ---
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            res_img = draw_ocr_results(img, det_results)
            cv2.imwrite(save_path, res_img)
            print(f"    Image saved to:  {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        print()

    print(f"=" * 60)
    print("DET Performance Data:")
    print(det_amlnn.get_perf_info())
    print("\nREC Performance Data:")
    print(rec_amlnn.get_perf_info())

    # Clean up
    det_amlnn.uninit()
    rec_amlnn.uninit()

if __name__ == "__main__":
    main()
"""
Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.

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
from shapely.geometry import Polygon
from pathlib import Path
from amlnn.api import AMLNN

# --- PPOCR-Det Configuration ---
MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 640
BOX_SCORE_THRESH = 0.5
BOX_THRESH = 0.3
UNCLIP_RATIO = 1.5
MIN_SIZE = 3
MAX_CANDIDATES = 1000

# Unnormalized ImageNet Mean & Std (Multiplied by 255)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def cv_point_compare(box):
    """Sort points by X coordinate"""
    return box[np.argsort(box[:, 0])]


def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=1.0, zp=0, tensor_type=2):
    """
    PPOCR Det Preprocess:
    1. Resize by ratio_max and pad bottom/right
    2. Normalize with unnormalized ImageNet Mean/Std
    3. Quantize using hardware scale (s), zero-point (zp), and tensor_type
    """
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 1. Resize and Pad (PPOCR Style - Top Left Alignment)
    h, w = rgb_img.shape[:2]
    width, height = new_shape

    ratio_max = max(w / width, h / height)
    new_w = min(int(w / ratio_max), width)
    new_h = min(int(h / ratio_max), height)

    resized_img = cv2.resize(rgb_img, (new_w, new_h))

    # Create padded image (black background)
    padded_img = np.zeros((height, width, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w, :] = resized_img

    # 2. Apply ImageNet Normalization Directly on [0, 255] range
    normalized_img = (padded_img.astype(np.float32) - MEAN) / STD

    # 3. Layout Formatting
    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    # 4. Hardware Quantization
    val = np.round(input_tensor / s + zp)
    
    if tensor_type == 2:   # INT8
        input_tensor = np.clip(val, -128, 127).astype(np.int8)
    elif tensor_type == 3: # UINT8
        input_tensor = np.clip(val, 0, 255).astype(np.uint8)
    # If tensor_type == 0 (float32), it passes through without clipping

    return input_tensor, original_img, ratio_max


def get_min_boxes(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    # Sort by X
    box = cv_point_compare(box)

    # Sort Y to find Top/Bottom
    if box[1, 1] > box[0, 1]:
        index1, index4 = 0, 1
    else:
        index1, index4 = 1, 0

    if box[3, 1] > box[2, 1]:
        index2, index3 = 2, 3
    else:
        index2, index3 = 3, 2

    min_box = np.array([box[index1], box[index2], box[index3], box[index4]], dtype=np.float32)

    min_side_len = min(rect[1][0], rect[1][1])
    perimeter = 2.0 * (rect[1][0] + rect[1][1])

    return min_box, min_side_len, perimeter


def get_box_score_fast(pred_map, box):
    box_pts = box.reshape(-1, 2)
    h, w = pred_map.shape

    min_x = int(np.min(box_pts[:, 0]))
    max_x = int(np.max(box_pts[:, 0]))
    min_y = int(np.min(box_pts[:, 1]))
    max_y = int(np.max(box_pts[:, 1]))

    max_x = min(max(max_x, 0), w - 1)
    max_y = min(max(max_y, 0), h - 1)
    min_x = max(min(min_x, w - 1), 0)
    min_y = max(min(min_y, h - 1), 0)

    local_box = box_pts.copy()
    local_box[:, 0] -= min_x
    local_box[:, 1] -= min_y

    mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [local_box.astype(np.int32)], 1)

    crop_map = pred_map[min_y:max_y+1, min_x:max_x+1].copy()
    score = cv2.mean(crop_map, mask=mask)[0]

    return score


def unclip(box, perimeter, unclip_ratio):
    poly = Polygon(box)
    if poly.length == 0:
        return np.array([], dtype=np.float32)

    distance = unclip_ratio * poly.area / perimeter

    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int32).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)

    out_box = []
    for path in expanded:
        for pt in path:
            out_box.append([pt[0], pt[1]])
            
    return np.array(out_box, dtype=np.float32)


def find_box(pred_map, bit_map, image_shape, scale):
    contours, _ = cv2.findContours(bit_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    num_contours = min(len(contours), MAX_CANDIDATES)
    res_boxes = []

    for i in range(num_contours):
        contour = contours[i]
        if len(contour) <= 2:
            continue

        min_box, min_side_len, perimeter = get_min_boxes(contour)

        if min_side_len < MIN_SIZE:
            continue

        score = get_box_score_fast(pred_map, contour)
        if score < BOX_SCORE_THRESH:
            continue

        clip_box = unclip(min_box, perimeter, UNCLIP_RATIO)
        if len(clip_box) == 0:
            continue

        clip_min_box, min_side_len2, perimeter2 = get_min_boxes(clip_box)
        if min_side_len2 < MIN_SIZE + 2:
            continue

        # Scale coordinates back to original image
        clip_min_box[:, 0] = np.clip(np.round(clip_min_box[:, 0] * scale), 0, image_shape[1])
        clip_min_box[:, 1] = np.clip(np.round(clip_min_box[:, 1] * scale), 0, image_shape[0])

        res_boxes.append({
            'box': clip_min_box.astype(np.int32),
            'score': float(score)
        })

    return res_boxes


def postprocess(outputs, original_shape, scale):
    # Dequantized output from API inference
    pred_map = np.squeeze(outputs[0])  # Shape becomes (H, W)

    # Binarize
    bit_map = (pred_map > BOX_THRESH).astype(np.uint8) * 255

    # Dilate Map
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bit_map = cv2.dilate(bit_map, kernel, iterations=1)

    return find_box(pred_map, bit_map, original_shape, scale)


def draw_detections(img, detections, save_path):
    result_img = img.copy()

    print(f"    Detected {len(detections)} text regions")

    for i, det in enumerate(detections):
        box = det['box']
        score = det['score']

        print(f"      {i+1}. Text Region (Score: {score:.2f})")
        cv2.polylines(result_img, [box], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imwrite(save_path, result_img)
    print(f"    Image saved to: {save_path}")
    return result_img


def main():
    parser = argparse.ArgumentParser(description="PPOCRv4 Det Demo")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    args = parser.parse_args()

    # 1. Initialize Runtime and Load Model
    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.adla)

    tensor_info = amlnn.get_tensor_info()
    print(amlnn.get_sdk_version())

    # 2. Find Images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files.sort()

    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        amlnn.uninit()
        return

    # 3. Retrieve Quantization Attributes
    tensor_attr = tensor_info["inputs"][0]
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    # 4. Generate Output Directory
    model_stem = Path(args.adla).stem
    result_dir = f"{model_stem}_result"
    os.makedirs(result_dir, exist_ok=True)

    # 5. Process loop
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess (Resizing, Normalizing, Quantizing)
            input_tensor, original_img, scale = preprocess(
                image_path,
                new_shape=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
                data_format='NHWC',
                s=s, zp=zp, tensor_type=tensor_type
            )

            # Inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess (Binarization, Contours, PyClipper)
            detections = postprocess(outputs, original_img.shape, scale)

            # Draw and Save
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")
            draw_detections(original_img, detections, str(save_path))

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        print()
        
    print(f"=" * 60)
    print(amlnn.get_perf_info())

    # Optional Visualization
    # amlnn.perf_visualize()

    # Clean up
    amlnn.uninit()

if __name__ == "__main__":
    main()
#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import os
import glob
import argparse
import cv2
from pathlib import Path
from amlnn.api import AMLNN
from PIL import Image, ImageDraw, ImageFont

LPR_MODEL_WIDTH = 94
LPR_MODEL_HEIGHT = 24

# LPRNet Standard Normalization
MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
STD  = np.array([128.0, 128.0, 128.0], dtype=np.float32)

# Standard Chinese LPRNet Dictionary (68 Classes)
LPR_CHARS = (
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
    "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", 
    "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", 
    "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", 
    "V", "W", "X", "Y", "Z", "学", "警", "-"
)
BLANK_IDX = len(LPR_CHARS) - 1  # 67 (The '-' token)


def preprocess(img_path, new_shape=(94, 24), data_format='NHWC', s=0.0, zp=0, tensor_type=None):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    # 1. Resize (This model takes in BGR)
    resized_img = cv2.resize(original_img, new_shape)
    # rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    normalized_img = (resized_img.astype(np.float32) - MEAN) / STD

    # 2. Strict Layout Formatting based on data_format
    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    if tensor_type in [2, 3] and s != 0:
        raw_val = np.round(input_tensor / s + zp)
        if tensor_type == 2:
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
    else:
        input_tensor = input_tensor.astype(np.float32) #FP16
            
    return input_tensor, original_img

def postprocess(outputs):
    """ LPRNet CTC Greedy Decoder """
    pred_map = np.squeeze(outputs[0])

    if len(pred_map.shape) < 2:
        return "", 0.0

    # Safety check: If shape is [68, 18], transpose it to [18, 68]
    if pred_map.shape[0] == len(LPR_CHARS):
        pred_map = pred_map.T 

    seq_len = pred_map.shape[0]

    text = ""
    total_score = 0.0
    valid_count = 0
    pre_idx = -1

    for i in range(seq_len):
        max_idx = int(np.argmax(pred_map[i]))
        max_score = float(pred_map[i][max_idx])

        # CTC Rules: Ignore blank token and consecutive duplicates
        if max_idx != BLANK_IDX and max_idx != pre_idx:
            if max_idx < len(LPR_CHARS):
                text += LPR_CHARS[max_idx]
            total_score += max_score
            valid_count += 1

        pre_idx = max_idx

    avg_score = total_score / valid_count if valid_count > 0 else 0.0
    return text, avg_score


def draw_detections(image, text, score):
    padding_top = 40
    padded_img = cv2.copyMakeBorder(image, padding_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Load a font capable of rendering Chinese Characters
    font = None
    font_paths = ["simhei.ttf", "simsun.ttc", "NotoSansCJK-Regular.ttc", "wqy-microhei.ttc"]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 24)
            break
        except IOError:
            continue

    if font is None:
        font = ImageFont.load_default()

    display_text = f"{text} ({score:.2f})"
    draw.text((10, 10), display_text, font=font, fill=(0, 255, 0))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="LPRNet (Chinese) Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing cropped license plates')
    args = parser.parse_args()

    # 1. Initialize Runtime and Load Model
    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)

    print(amlnn.get_sdk_version())

    tensor_info = amlnn.get_tensor_info()

    tensor_attr = tensor_info["inputs"][0]
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files.sort()

    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        amlnn.uninit()
        return

    model_stem = Path(args.model_path).stem
    result_dir = f"{model_stem}_result"
    os.makedirs(result_dir, exist_ok=True)

    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess
            input_tensor, original_img = preprocess(
                image_path,
                new_shape=(LPR_MODEL_WIDTH, LPR_MODEL_HEIGHT),
                data_format='NHWC', 
                s=s, zp=zp, tensor_type=tensor_type
            )

            # Inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess (CTC Decoder)
            text, score = postprocess(outputs)
            print(f"    Recognized Plate: [{text}]")

            # Save Image
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            result_img = draw_detections(original_img, text, score)
            cv2.imwrite(save_path, result_img)
            print(f"    Image saved to:  {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        print()

    print(f"=" * 60)
    print(amlnn.get_perf_info())
    amlnn.uninit()

if __name__ == "__main__":
    main()
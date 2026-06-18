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
from pathlib import Path
from amlnn.api import AMLNN
from PIL import Image, ImageDraw, ImageFont

REC_MODEL_WIDTH = 320
REC_MODEL_HEIGHT = 48

MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
STD  = np.array([127.5, 127.5, 127.5], dtype=np.float32)


def load_dictionary(dict_path):
    """Loads the PP-OCR dictionary to map output indices to characters."""
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

    dictionary = ['blank']
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            dictionary.append(line.strip('\r\n'))

    dictionary.append(' ')
    return dictionary


def preprocess(img_path, new_shape=(320, 48), data_format='NHWC', s=1.0, zp=0, tensor_type=2):
    """
    PPOCR Rec Preprocess:
    1. Aspect-ratio preserving resize and pad right with black.
    2. Normalize: (img - 127.5) / 127.5
    3. Quantize using hardware scale (s), zero-point (zp), and tensor_type.
    """
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = rgb_img.shape[:2]
    width, height = new_shape

    # 1. Aspect-ratio preserving resize
    ratio = orig_w / orig_h
    new_w = min(int(height * ratio), width)
    new_w = max(1, new_w) 

    resized_img = cv2.resize(rgb_img, (new_w, height))

    # Pad the rest of the width with 0 (Black)
    padded_img = np.zeros((height, width, 3), dtype=np.uint8)
    padded_img[0:height, 0:new_w] = resized_img

    # 2. Normalization
    padded_img = padded_img.astype(np.float32)
    normalized_img = (padded_img - MEAN) / STD

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
    elif tensor_type == 4: # INT16
        input_tensor = np.clip(val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}.")

    return input_tensor, original_img


def postprocess(outputs, dictionary):
    """ CTC Greedy Decoder """
    pred_map = np.squeeze(outputs[0])

    if len(pred_map.shape) < 2:
        return "", 0.0

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
            if max_idx < len(dictionary):
                text += dictionary[max_idx]
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

    font = ImageFont.load_default()

    display_text = f"Result: {text} | Conf: {score:.3f}"
    draw.text((10, 5), display_text, font=font, fill=(0, 255, 0))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="PPOCR Rec Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--dict-path', required=True, help='Path to PP-OCR dictionary.txt file')
    args = parser.parse_args()

    print("PPOCR Rec Demo")

    # 1. Load Dictionary
    char_dict = load_dictionary(args.dict_path)

    # 2. Initialize Runtime and Load Model
    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()
    print(amlnn.get_sdk_version())

    # 3. Retrieve Quantization Attributes
    tensor_attr = tensor_info["inputs"][0]
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    # 4. Find Images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    image_files.sort()

    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        amlnn.uninit()
        return

    # 5. Generate Output Directory
    model_stem = Path(args.model_path).stem
    result_dir = f"{model_stem}_result"
    os.makedirs(result_dir, exist_ok=True)

    # 6. Process loop
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess (Resizing, Normalizing, Quantizing)
            input_tensor, original_img = preprocess(
                image_path,
                new_shape=(REC_MODEL_WIDTH, REC_MODEL_HEIGHT),
                data_format='NHWC',
                s=s, zp=zp, tensor_type=tensor_type
            )

            # Inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess (CTC Decoder)
            text, score = postprocess(outputs, dictionary=char_dict)

            print(f"    Recognized Text: [{text}]")

            # Draw and Save
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

    # Optional Visualization
    # amlnn.perf_visualize()

    # Clean up
    amlnn.uninit()

if __name__ == "__main__":
    main()
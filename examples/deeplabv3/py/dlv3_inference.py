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

MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)
STD  = np.array([58.395, 57.120, 57.375], dtype=np.float32)

# DeepLabV3 21 Classes typically correspond to Pascal VOC
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

def get_pascal_colors(num_classes=21):
    """Generates the standard Pascal VOC color palette for segmentation masks."""
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r, g, b = 0, 0, 0
        id_ = i
        for j in range(8):
            if id_ & (1 << 0): r |= (1 << (7 - j))
            if id_ & (1 << 1): g |= (1 << (7 - j))
            if id_ & (1 << 2): b |= (1 << (7 - j))
            id_ >>= 3
        colors[i] = [b, g, r] # BGR format for OpenCV blending
    return colors

def letterbox(img, new_shape=(512, 512), color=(114, 114, 114)):
    shape = img.shape[:2]  # [height, width]
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
    
    pad_w = (new_shape[1] - new_unpad[0]) / 2
    pad_h = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, scale, (left, top), (new_unpad[0], new_unpad[1])

def preprocess(img_path, new_shape=(512, 512), data_format='NCHW', s=0.003789, zp=-128, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad, unpad_shape = letterbox(original_img, new_shape)

    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    # 1. Cast to float32
    img_float = rgb_img.astype(np.float32) 

    # 2. Subtract Mean and Divide by Std
    normalized_img = (img_float - MEAN) / STD

    # 3. Handle Layout Mapping
    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    # 4. AMLNN Quantization Logic
    # 0 = Float32/FP16, 2 = INT8, 3 = UINT8
    if tensor_type == 2:
        raw_val = np.round(input_tensor / s + zp)
        input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        raw_val = np.round(input_tensor / s + zp)
        input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
    elif tensor_type == 0:
        # Float32 / FP16 bypasses quantization
        input_tensor = input_tensor.astype(np.float32)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")

    return input_tensor, original_img, scale, pad, unpad_shape

def postprocess(outputs, original_img_shape, pad, unpad_shape, data_format='NCHW'):
    # Extract the main logit output
    logits = outputs[0] 

    # Argmax over the class channels to find the predicted class per pixel
    # NCHW layout means channels are at axis=1 -> shape becomes [1, 512, 512]
    if data_format == 'NCHW':
        mask_idx = np.argmax(logits, axis=1)
    else:
        mask_idx = np.argmax(logits, axis=-1)

    # Drop the batch dimension -> [512, 512]
    mask_2d = mask_idx[0].astype(np.uint8)

    # Extract padding offsets and unpadded shape
    left, top = pad
    unpad_w, unpad_h = unpad_shape

    # Crop the valid area out of the padded 512x512 mask
    valid_mask = mask_2d[top : top + unpad_h, left : left + unpad_w]

    # Resize the valid mask back to the original image dimensions.
    orig_h, orig_w = original_img_shape[:2]
    final_mask = cv2.resize(valid_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return final_mask

def draw_segmentation(img, mask, save_path):
    colors = get_pascal_colors(len(VOC_CLASSES))
    color_mask = colors[mask]

    alpha = 0.6  
    blended_img = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

    cv2.imwrite(save_path, blended_img)
    return blended_img

def main():
    parser = argparse.ArgumentParser(description="DeepLabV3 Demo")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native", enable_perf=True)
    amlnn.load_model(path=args.adla)

    tensor_info = amlnn.get_tensor_info()

    tensor_attr = tensor_info["inputs"][0]
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    print(amlnn.get_sdk_version())

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return 0

    print(f"Model expects scale={s}, zero_point={zp}, type={tensor_type}")
    print(f"Found {len(image_files)} image file(s) to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # 1. Preprocess with Quantization mapping applied
            input_tensor, original_img, scale, pad, unpad_shape = preprocess(
                image_path, 
                new_shape=(512, 512), 
                data_format='NHWC',
                s=s, 
                zp=zp, 
                tensor_type=tensor_type
            )

            # 2. Run inference
            outputs = amlnn.inference(
                inputs=[input_tensor]
            )

            # 3. Postprocess results
            mask = postprocess(outputs, original_img.shape, pad, unpad_shape, data_format='NHWC')

            # Identify found classes for logging
            unique_classes = np.unique(mask)
            found_labels = [VOC_CLASSES[cls_id] for cls_id in unique_classes if cls_id != 0] 
            
            if found_labels:
                print(f"    Detected classes: {', '.join(found_labels)}")
            else:
                print("    No foreground objects detected")

            # 4. Save result image
            model_name = Path(args.adla).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            draw_segmentation(original_img, mask, str(save_path))
            print(f"    Result saved to: {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()
        
    print(f"=" * 60)
    print(amlnn.get_perf_info())

    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
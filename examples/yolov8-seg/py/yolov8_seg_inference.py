"""
Copyright (C) 2026 Amlogic, Inc. All rights reserved.

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

MEAN = np.array([0, 0, 0], dtype=np.float32)
STD  = np.array([255, 255, 255], dtype=np.float32)

class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'doughnut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
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

    return img, scale, (left, top)

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC', s=0.003789, zp=-128, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0

    if data_format == 'NCHW':
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}.")

    raw_val = np.round(input_tensor / s + zp)
    if tensor_type == 2:
        input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)

    return input_tensor, original_img, scale, pad


def postprocess(outputs, scale, pad, strides=[8, 16, 32], conf_threshold=0.25, iou_threshold=0.7):
    # 1. Extract and shape properly

    # Bounding boxes
    bboxes = np.squeeze(outputs[0])      # (4, 8400)
    bboxes = bboxes.T                    # (8400, 4)

    # Class scores
    class_scores = np.squeeze(outputs[1])   # (80, 8400)
    class_scores = class_scores.T           # (8400, 80)

    # Mask coefficients
    mask_coeff = np.squeeze(outputs[2])      # (32, 8400)
    mask_coeff = mask_coeff.T                 # (8400, 32)

    # Prototype masks
    proto_mask = np.squeeze(outputs[3])     # (160, 160, 32)
    proto_mask = proto_mask.transpose(2,0,1) # (32, 160, 160)

    # 2. Filter out low confidence detections
    max_scores = np.max(class_scores, axis=1)
    mask = max_scores > conf_threshold

    filtered_bboxes = bboxes[mask]
    filtered_class_scores = class_scores[mask]
    filtered_mask_coeff = mask_coeff[mask]

    if len(filtered_bboxes) == 0:
        return []

    # 3. Process Bounding Boxes (cx, cy, w, h -> x1, y1, x2, y2)
    cx, cy, w, h = filtered_bboxes[:, 0], filtered_bboxes[:, 1], filtered_bboxes[:, 2], filtered_bboxes[:, 3]

    pad_x, pad_y = pad
    x1 = (cx - w/2 - pad_x) / scale
    y1 = (cy - h/2 - pad_y) / scale
    x2 = (cx + w/2 - pad_x) / scale
    y2 = (cy + h/2 - pad_y) / scale
    w /= scale
    h /= scale

    boxes_xyxy_scaled = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xywh_scaled = np.stack([x1, y1, w, h], axis=1)

    # 5. NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh_scaled.tolist(),
        np.max(filtered_class_scores, axis=1).tolist(),
        conf_threshold,
        iou_threshold
    )

    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detections.append({
                'bbox': boxes_xyxy_scaled[idx].tolist(),
                'class_id': np.argmax(filtered_class_scores[idx]),
                'score': float(np.max(filtered_class_scores[idx])),
                'mask_coeff': filtered_mask_coeff[idx].tolist()
            })

    return detections, proto_mask


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_mask(img, mask_coeff, proto_mask, scale, pad, color, alpha=0.5):
    """
    Draws mask on the image
    """
    orig_h, orig_w = img.shape[:2]
    pad_x, pad_y = pad

    # 1. Build raw mask from prototypes
    mask = np.tensordot(mask_coeff, proto_mask, axes=1)
    mask = sigmoid(mask)

    # 2. Upscale mask to the model input size (640x640)
    input_size = 640
    mask_640 = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # 3. Crop out the letterbox padding
    y1, y2 = int(pad_y), int(input_size - pad_y)
    x1, x2 = int(pad_x), int(input_size - pad_x)

    # Safety check to ensure we don't crop outside bounds
    y1, y2 = max(0, y1), min(input_size, y2)
    x1, x2 = max(0, x1), min(input_size, x2)

    mask_cropped = mask_640[y1:y2, x1:x2]

    # 4. Resize to match original image dimensions
    final_mask = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 5. Create a Boolean mask
    # Values > 0.5 are the object, < 0.5 is background
    binary_mask = final_mask > 0.5

    # 6. Create an image-sized green overlay
    colored_mask = np.zeros_like(img, dtype=np.uint8)
    colored_mask[binary_mask] = color # Green in BGR

    # Blend the FULL images together
    blended = cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)

    # Only update the original image where the mask exists
    result = img.copy()
    result[binary_mask] = blended[binary_mask]

    return result

def get_class_color(class_id):
    import colorsys
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    return bgr


def draw_detections(img, detections, proto_mask, scale, pad, save_path):
    result_img = img.copy()

    print(f"    Drawing {len(detections)} detections")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        class_id = det['class_id']
        score = det['score']
        mask_coeff = det['mask_coeff']

        print(f"    Detection {i+1}:")
        print(f"      BBox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"      Score: {score:.4f}")

        color = get_class_color(class_id)

        # Draw Bounding Box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        # Draw Label
        label = f"{class_names[class_id]}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        result_img = draw_mask(result_img, mask_coeff, proto_mask, scale, pad, color=color)

    cv2.imwrite(save_path, result_img)
    print(f"    Image saved to: {save_path}")
    return result_img


def main():
    parser = argparse.ArgumentParser(description="Yolov8-seg Demo")
    parser.add_argument('--model-path', required=True, help='Path to .adla model')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.init_runtime(mode="native", enable_perf=True)

    amlnn.load_model(path=args.model_path)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return 0

    print(f"Found {len(image_files)} image file(s) to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    tensor_attr = tensor_info["inputs"][0]
    s = float(tensor_attr["scale"])
    zp = int(tensor_attr["zp"])
    tensor_type = int(tensor_attr["type"])

    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            input_tensor, original_img, scale, pad = preprocess(
                image_path, new_shape=(640, 640), data_format = "NHWC", s=s, zp=zp, tensor_type=tensor_type
            )

            outputs = amlnn.inference(inputs=[input_tensor])

            detections, proto_mask = postprocess(outputs, scale, pad, conf_threshold=0.25, iou_threshold=0.45)

            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            draw_detections(original_img, detections, proto_mask, scale, pad, str(save_path))

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()
        
    print(f"=" * 60)
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
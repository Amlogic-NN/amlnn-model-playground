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

KEYPOINT_NAMES = [
    "nose","l_eye","r_eye","l_ear","r_ear",
    "l_sh","r_sh","l_el","r_el","l_wr","r_wr",
    "l_hip","r_hip","l_kn","r_kn","l_an","r_an"
]

# -----------------------------
# Skeleton
# -----------------------------
SKELETON = [
    (0,1),(0,2),
    (1,3),(2,4),
    (5,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

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


def postprocess(outputs, scale, pad, data_format='NCHW', strides=[8, 16, 32], conf_threshold=0.25, iou_threshold=0.45):
    # 1. Extract and shape properly
    bboxes = np.squeeze(outputs[0])  # Shape: (4, 8400)
    bboxes = bboxes.T            # Shape becomes: (8400, 4)
    conf = np.squeeze(outputs[1])  # Shape: (8400,)
    kpts_conf = np.squeeze(outputs[2])  # Shape: (8400, 17)
    kpts_xy = np.squeeze(outputs[3])    # Shape: (2, 8400, 17)
    kpts_xy = kpts_xy.transpose(1, 2, 0)  # Shape becomes: (8400, 17, 2)

    # 2. Filter out low confidence detections
    mask = conf > conf_threshold

    filtered_bboxes = bboxes[mask]
    filtered_conf = conf[mask]
    filtered_kpts_xy = kpts_xy[mask]
    filtered_kpts_conf = kpts_conf[mask]

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
    filtered_kpts_xy[:, :, 0] = (filtered_kpts_xy[:, :, 0] - pad_x) / scale
    filtered_kpts_xy[:, :, 1] = (filtered_kpts_xy[:, :, 1] - pad_y) / scale

    # 5. NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh_scaled.tolist(),
        filtered_conf.tolist(),
        conf_threshold,
        iou_threshold
    )

    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            detections.append({
                'bbox': boxes_xyxy_scaled[idx].tolist(),
                'confidence': float(filtered_conf[idx]),
                'keypoints': filtered_kpts_xy[idx].tolist(),
                'kptconfidence': filtered_kpts_conf[idx].tolist()
            })
    return detections


def draw_pose(img, keypoints, keypoints_conf):
    img_height, img_width = img.shape[:2]

    # draw points + labels
    for i, (x, y) in enumerate(keypoints):
        if keypoints_conf[i] < 0.5:
            continue

        if x < 0 or x >= img_width or y < 0 or y >= img_height:
            continue

        x, y = int(x), int(y)

        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        cv2.putText(
            img,
            KEYPOINT_NAMES[i],
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

    # draw skeleton
    for a, b in SKELETON:
        if keypoints_conf[a] > 0.5 and keypoints_conf[b] > 0.5:
            x1, y1 = keypoints[a][0], keypoints[a][1]
            x2, y2 = keypoints[b][0], keypoints[b][1]

            if (0 <= x1 < img_width and 0 <= y1 < img_height and
                0 <= x2 < img_width and 0 <= y2 < img_height):

                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def get_color(conf):
    import colorsys
    hue = (conf * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    return bgr


def draw_detections(img, detections, save_path):
    result_img = img.copy()

    print(f"    Drawing {len(detections)} detections")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        confidence = det['confidence']
        keypoints = det['keypoints']
        kptsconf = det['kptconfidence']

        print(f"    Detection {i+1}:")
        print(f"      BBox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"      Confidence: {confidence:.4f}")

        img_height, img_width = img.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            print(f"      WARNING: BBox partly outside image bounds!")

        visible_kpts = sum(1 for c in kptsconf if c > 0.5)
        print(f"      Visible keypoints: {visible_kpts}/17")

        color = get_color(confidence)

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

        label = f"conf: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Draw pose keypoints
        draw_pose(result_img, keypoints, kptsconf)

    cv2.imwrite(save_path, result_img)
    print(f"    Image saved to: {save_path}")
    return result_img


def main():
    parser = argparse.ArgumentParser(description="Yolov8-pose Demo")
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
                image_path, new_shape=(640, 640), data_format='NHWC', s=s, zp=zp, tensor_type=tensor_type
            )

            outputs = amlnn.inference(inputs=[input_tensor])

            detections = postprocess(outputs, scale, pad, conf_threshold=0.25, iou_threshold=0.45)

            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")

            draw_detections(original_img, detections, str(save_path))

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print(f"=" * 60)    
    print(amlnn.get_perf_info())
    amlnn.perf_visualize()
    amlnn.uninit()

if __name__ == "__main__":
    main()
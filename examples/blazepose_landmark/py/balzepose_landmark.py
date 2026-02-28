#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
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
from amlnnlite.api import AMLNNLite
import math

def preprocess(img_path, detections, new_shape=(256, 256), data_format='NCHW', s=0.003921568859368563, zp=-128):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")
    
    im_h, im_w, _ = original_img.shape
    if detections.shape[0] > 0:
        detections = detections[:1, :]  
    else:
        raise ValueError("No detections input, please run blazepose_detect and generate the detections first.")

    x_center, y_center = detections[4:6]
    x_scale, y_scale = detections[6:8]
    print(f"---------center {x_center}, {y_center}, x_scale {x_scale}, y_scale {y_scale}")

    box_size = (((x_scale - x_center) ** 2 + (y_scale - y_center) ** 2) ** 0.5) * 2
    box_size *= 1.25

    angle = (np.pi * 90 / 180) - math.atan2(-(y_scale - y_center), x_scale - x_center)
    rotation = angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))

    rotated_rect = ((x_center, y_center), (box_size, box_size), rotation * 180. / np.pi)
    pts1 = cv2.boxPoints(rotated_rect)

    h, w = new_shape
    pts2 = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    processed_img = cv2.warpPerspective(original_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0

    if data_format == 'NCHW':
        # HWC -> CHW -> BCHW (ONNX default format)
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        # HWC -> BHWC (TFLITE default format)
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Only 'NCHW' and 'NHWC' are supported.")
    
    # Quantize to int16
    input_tensor = np.round(input_tensor / s + zp).astype(np.int16)
    
    return input_tensor, original_img, [x_center, y_center, rotation, box_size]

def tensor_to_landmark(landmarks):
    num_landmarks = 39
    num_dimensions = landmarks.shape[1] # num_landmarks
    output = landmarks.reshape(-1, num_landmarks, num_dimensions).copy()
    if num_dimensions > 3:
        output[..., 3:5] = 1.0 / (1.0 + np.exp(-output[..., 3:5]))
    return output

def refine_landmark(landmarks, heatmap):#39*5, 64*64*39
    min_confidence = 0.5
    kernel_size = 9
    offset = kernel_size   

    hm_h, hm_w, _ = heatmap.shape

    for i, lm in enumerate(landmarks):
        col = int(lm[0] * hm_w)
        row = int(lm[1] * hm_h)

        if not (0 <= col < hm_w and 0 <= row < hm_h):
            continue

        c0 = max(0, col - offset)
        c1 = min(hm_w, col + offset + 1)
        r0 = max(0, row - offset)
        r1 = min(hm_h, row + offset + 1)

        val_sum = 0.0
        weighted_col = 0.0
        weighted_row = 0.0
        max_conf = 0.0

        for r in range(r0, r1):
            for c in range(c0, c1):
                conf = 1.0 / (1.0 + np.exp(-heatmap[r, c, i]))
                val_sum += conf
                max_conf = max(max_conf, conf)
                weighted_col += c * conf
                weighted_row += r * conf

        if max_conf >= min_confidence and val_sum > 0:
            lm[0] = weighted_col / (hm_w * val_sum)
            lm[1] = weighted_row / (hm_h * val_sum)

    return landmarks

def postprocess(outputs, params, data_format='NCHW'):
    x_center, y_center, rotation, box_size = params
    flag, landmark_tensor, world_landmark_tensor, segment, heatmap_tensor = [], [], [], [], []
    for out in outputs:
        if len(out.shape) == 2:
            if out.shape == (1, 1) or out.shape == (1,):
                flag = out
            elif out.shape[1] == 195:
                landmark_tensor = out
            elif out.shape[1] == 117:
                world_landmark_tensor = out
        elif len(out.shape) == 4 and out.shape[3] == 1 and out.shape[1] == 256:
            segment = out
        elif len(out.shape) == 4 and out.shape[1] == 64 and out.shape[3] == 39:
            heatmap_tensor = out

    raw_landmarks = tensor_to_landmark(landmark_tensor)
    all_world_landmarks = tensor_to_landmark(world_landmark_tensor)

    h = w = 256
    raw_landmarks[:, :, 0] = raw_landmarks[:, :, 0] / w
    raw_landmarks[:, :, 1] = raw_landmarks[:, :, 1] / h
    raw_landmarks[:, :, 2] = raw_landmarks[:, :, 2] / w

    # Refines landmarks with the heatmap tensor.
    all_landmarks = refine_landmark(raw_landmarks[0], heatmap_tensor[0])
    all_world_landmarks = all_world_landmarks[0]

    print(f"rotation {rotation}")
    cosa = math.cos(rotation)
    sina = math.sin(rotation)
    for landmark in all_landmarks:
        x = landmark[0] - 0.5
        y = landmark[1] - 0.5
        landmark[0] = ((cosa * x - sina * y) * box_size + x_center) 
        landmark[1] = ((sina * x + cosa * y) * box_size + y_center)  
        landmark[2] = landmark[2] * box_size 
        
    # Projects the world landmarks from the letterboxed ROI to the full image.
    for landmark in all_world_landmarks:
        x = landmark[0]
        y = landmark[1]
        landmark[0] = cosa * x - sina * y
        landmark[1] = sina * x + cosa * y
    
    return all_landmarks

def get_class_color(class_id):
    import colorsys
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    return bgr

POSE_CONNECTIONS = [
    # Face
    (0, 1),(1, 2),(2, 3),(3, 7),
    (0, 4),(4, 5),(5, 6),(6, 8),
    # Mouth
    (9, 10),
    # Shoulders
    (11, 12),
    # Right arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Right leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

def draw_landmarks(img, landmarks, save_path, score_threshold=0.5):
    result_img = img.copy()
    for lm in landmarks:
        lms = lm.landmarks   

        for point in lms:
            x, y, score = int(point[0]), int(point[1]), point[3]
            if score < score_threshold:
                continue
            cv2.circle(result_img, (x, y), 3, (0, 255, 0), -1)

        for i0, i1 in POSE_CONNECTIONS:
            if i0 >= len(lms) or i1 >= len(lms):
                continue
            if lms[i0][3] < score_threshold or lms[i1][3] < score_threshold:
                continue
            p0 = (int(lms[i0][0]), int(lms[i0][1]))
            p1 = (int(lms[i1][0]), int(lms[i1][1]))
            cv2.line(result_img, p0, p1, (255, 0, 0), 2)

    cv2.imwrite(save_path, result_img)
    return result_img

def read_detections_from_txt(txt_path):
    with open(txt_path, "r") as f:
        detections = [[float(x) for x in line.split()] for line in f if line.strip()]
    return np.array(detections, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='./blazepose_landmark_int8_A311D2.adla')
    parser.add_argument('--run-cycles', default= 1, type=int)
    args = parser.parse_args()
    
    # Initialize AMLNNLite
    amlnn = AMLNNLite()
    amlnn.config(
        model_path=args.model_path,           # Model file path, Support ADLA and quantized TFlite models
        run_cycles=args.run_cycles
    )
    amlnn.init()

    # Find all image files in the 01_export_model directory
    image_dir = "./"
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print("No image files found in", image_dir)
        amlnn.uninit()
        return
    
    print(f"Found {len(image_files)} image files to process:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    print()

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        detections = read_detections_from_txt(txt_path=txt_path)
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)
        
        try:
            # Preprocess input
            input_tensor, original_img, params = preprocess(image_path, detections, new_shape=(256, 256), data_format='NHWC', s=0.000030518509447574615, zp=0)

            # Run inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess results
            landmarks = postprocess(outputs, params, data_format='NHWC')
            
            # Print detection results
            if landmarks:
                print(f"    Detected {len(landmarks)} objects:")
                for i, lm in enumerate(landmarks, 1):
                    print(f"      {i}. {lm['class_name']} ({lm['confidence']:.2f})")
            else:
                print("    No objects detected")
            
            # Save result image
            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")
            draw_landmarks(original_img, landmarks, str(save_path), score_threshold=0.5)
            print(f"    Result saved to: {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
        
        print()

    # Optional visualization
    amlnn.visualize()

    # Release resources
    amlnn.uninit()

if __name__ == "__main__":
    main()


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
from amlnnlite.api import AMLNNLite

def letterbox(img, new_shape=(224, 224), color=(0, 0, 0)):
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
    
    scale = 1. / scale
    ori_left = left * scale 
    ori_top = top * scale
    return img, scale, (ori_left, ori_top)

def preprocess(img_path, new_shape=(224, 224), data_format='NCHW', s=0.003921568859368563, zp=-128):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")
    
    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 127.5 - 1.

    if data_format == 'NCHW':
        # HWC -> CHW -> BCHW (ONNX default format)
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        # HWC -> BHWC (TFLITE default format)
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Only 'NCHW' and 'NHWC' are supported.")
    
    # Quantize to int8
    input_tensor = np.round(input_tensor / s + zp).astype(np.int8)
    
    return input_tensor, original_img, scale, pad


def postprocess(outputs, scale, pad, data_format='NCHW', anchor_path='anchors.npy', score_threshold=0.5, nms_threshold=0.3):
    all_boxes = []
    all_scores = []

    raw_box = outputs[0]   # (1, 2254, 12)
    raw_score = outputs[1] # (1, 2254, 1)

    anchors = np.load(anchor_path).astype("float32")

    # all_boxes = decode_boxes(raw_box, anchors)
    # anchors: [N, 4] -> x, y, w, h
    anc_x, anc_y, anc_w, anc_h = anchors.T
    # raw_box shape: [..., K]
    all_boxes = np.zeros_like(raw_box)
    # box center & size
    x_center = raw_box[..., 0] / 224.0 * anc_w + anc_x
    y_center = raw_box[..., 1] / 224.0 * anc_h + anc_y
    w = raw_box[..., 2] / 224.0 * anc_w
    h = raw_box[..., 3] / 224.0 * anc_h
    # bbox: ymin, xmin, ymax, xmax
    all_boxes[..., 0] = y_center - 0.5 * h
    all_boxes[..., 1] = x_center - 0.5 * w
    all_boxes[..., 2] = y_center + 0.5 * h
    all_boxes[..., 3] = x_center + 0.5 * w
    # keypoints (4 points, each has x/y)
    for k in range(4):
        idx = 4 + k * 2
        all_boxes[..., idx]     = raw_box[..., idx]     / 224.0 * anc_w + anc_x
        all_boxes[..., idx + 1] = raw_box[..., idx + 1] / 224.0 * anc_h + anc_y
    

    thresh = 100.0
    raw_score = raw_score.clip(-thresh, thresh)
    # Apply sigmoid activation to class scores
    all_scores = 1.0 / (1.0 + np.exp(-raw_score)).squeeze(axis=-1)
    print(f"all_scores {all_scores}")
    print(f"max(all_scores) {max(all_scores[0])}")

    mask = all_scores >= score_threshold

    # Merge all scales
    final_boxes = np.concatenate(all_boxes, axis=0)
    final_scores = np.concatenate(all_scores, axis=0)

    # Filter by confidence threshold
    valid_mask = final_scores > score_threshold
    if not np.any(valid_mask):
        return []
    
    valid_boxes = final_boxes[valid_mask]
    valid_scores = final_scores[valid_mask]

    # Map coordinates back to original image
    pad_x, pad_y = pad
    s = scale * 224
    valid_boxes[:, [0, 2]] = valid_boxes[:, [0, 2]] * s - pad_x
    valid_boxes[:, [1, 3]] = valid_boxes[:, [1, 3]] * s - pad_y
    valid_boxes[:, 4::2] = valid_boxes[:, 4::2] * s - pad_y
    valid_boxes[:, 5::2] = valid_boxes[:, 5::2] * s - pad_x
    valid_boxes = np.maximum(valid_boxes, 0)

    # NMS
    if len(valid_boxes) > 0:
        nms_indices = cv2.dnn.NMSBoxes(
            valid_boxes.tolist(), valid_scores.tolist(), score_threshold, nms_threshold
        )
        
        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            detections = []
            
            for idx in nms_indices:
                x1, y1, x2, y2 = valid_boxes[idx, :4]
                confidence = valid_scores[idx]

                # x_center = (valid_boxes[:,1] + valid_boxes[:,3]) / 2
                # y_center = (valid_boxes[:,0] + valid_boxes[:,2]) / 2
                # scale = (valid_boxes[:,3] - valid_boxes[:,1]) # assumes square boxes
                                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence)
                })
            
            return detections
    
    return []

def get_class_color(class_id):
    import colorsys
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    return bgr

def draw_detections(img, detections, save_path):
    result_img = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        confidence = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']
        
        color = get_class_color(class_id)
        
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
    
    cv2.imwrite(save_path, result_img)
    return result_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='./blazepose_detect_int8_A311D2.adla')
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
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)
        
        try:
            # Preprocess input
            input_tensor, original_img, scale, pad = preprocess(image_path, new_shape=(224, 224), data_format='NHWC', s=0.007843137718737125, zp=-1)

            # Run inference
            outputs = amlnn.inference(inputs=[input_tensor])

            # Postprocess results
            detections = postprocess(outputs, scale, pad, data_format='NHWC', score_threshold=0.5, nms_threshold=0.3)
            
            # Print detection results
            if detections:
                print(f"    Detected {len(detections)} objects:")
                for i, det in enumerate(detections, 1):
                    print(f"      {i}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("    No objects detected")
            
            # Save result image
            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")
            draw_detections(original_img, detections, str(save_path))
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


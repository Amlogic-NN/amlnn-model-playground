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

NUM_COORDS = 12
MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
STD  = np.array([127.5, 127.5, 127.5], dtype=np.float32)

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
    padw = left * scale
    padh = top * scale
    return img, scale, (padh, padw)

def preprocess(img_path, new_shape=(224, 224), data_format='NCHW', s=0.007703, zp=1, tensor_type=2):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad = letterbox(original_img, new_shape)
    print(f"---------------scale {scale}, pad {pad}")

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
    if tensor_type == 2:
        input_tensor = np.round(input_tensor / s + zp).astype(np.int8)
    elif tensor_type == 3:
        input_tensor = np.round(input_tensor / s + zp).astype(np.uint8)
    print(f"---------------input_tensor {input_tensor.shape} {np.max(input_tensor)} {np.min(input_tensor)}")
    return input_tensor, original_img, scale, pad

def weighted_nms(boxes, scores, iou_thresh=0.3):
    x1, y1, x2, y2 = boxes.T
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / ((x2[i]-x1[i])*(y2[i]-y1[i]) + 1e-8)

        indices = indices[1:][iou <= iou_thresh]
    return np.array(keep)

def postprocess(outputs, scale, pad, data_format='NCHW', anchor_path='anchors.npy', score_threshold=0.5, nms_threshold=0.3):
    all_boxes = []
    all_scores = []

    raw_box = outputs[0]   # (1, 2254, 12)
    raw_score = outputs[1] # (1, 2254, 1)

    print(f"raw_box {raw_box.shape} {np.max(raw_box)} {np.min(raw_box)}")
    print(f"raw_score {raw_score.shape} {np.max(raw_score)} {np.min(raw_score)}")
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
    print(f"all_scores {all_scores.shape} {np.max(all_scores)} {np.min(all_scores)}")

    # print(f"max(all_scores) {max(all_scores[0])}")

    # mask = all_scores >= score_threshold

    # Merge all scales
    final_boxes = np.concatenate(all_boxes, axis=0)
    final_scores = np.concatenate(all_scores, axis=0)
    print(f"final_scores {final_scores.shape} {np.max(final_scores)} {np.min(final_scores)}")
    # Filter by confidence threshold
    valid_mask = final_scores >= score_threshold
    print(f"valid_mask {valid_mask.shape} {np.sum(valid_mask==1)}")
    if not np.any(valid_mask):
        return [], []

    valid_boxes = final_boxes[valid_mask]
    valid_scores = final_scores[valid_mask]

    # Map coordinates back to original image
    padh, padw = pad
    s = scale * 224
    valid_boxes[:, [0, 2]] = valid_boxes[:, [0, 2]] * s - padh
    valid_boxes[:, [1, 3]] = valid_boxes[:, [1, 3]] * s - padw
    valid_boxes[:, 4::2] = valid_boxes[:, 4::2] * s - padw
    valid_boxes[:, 5::2] = valid_boxes[:, 5::2] * s - padh
    valid_boxes = np.maximum(valid_boxes, 0)

    # NMS
    if len(valid_boxes) > 0:

        boxes = valid_boxes[:, :4]
        nms_indices = weighted_nms(boxes, valid_scores, iou_thresh=nms_threshold)
        print("nms_indices:", nms_indices)

        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            detections = []
            detections_show = []

            for idx in nms_indices:
                y1, x1, y2, x2 = valid_boxes[idx, :4]  ##convert yxyx to xyxy
                confidence = valid_scores[idx]

                # x_center = (valid_boxes[:,1] + valid_boxes[:,3]) / 2
                # y_center = (valid_boxes[:,0] + valid_boxes[:,2]) / 2
                # scale = (valid_boxes[:,3] - valid_boxes[:,1]) # assumes square boxes

                detections_show.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_name': "pose",
                    'confidence': float(confidence),
                    'class_id': idx,
                })
            detections.append( list(valid_boxes[idx]) + [valid_scores[idx]] )
            return detections, detections_show

    return [], []

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
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--dataset-path', required=True, help='Path to quant dataset')
    parser.add_argument('--image-dir', required=True, help='Directory containing test images')
    parser.add_argument('--target-platform', required=True, help='Platform ID, e.g. 001, 002, 003')
    args = parser.parse_args()

    # Initialize AMLNN
    amlnn = AMLNN()

    amlnn.load_tflite(model=args.model_path)

    amlnn.config(normalization_mean=[MEAN.tolist()], normalization_std=[STD.tolist()], quantized_dtype='w8a8', target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}", quantized_method='perchannel')

    amlnn.compile(dataset=args.dataset_path)

    amlnn.export_adla()

    amlnn.init_runtime(mode="native", enable_perf=True)

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

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"=" * 60)
        print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"=" * 60)

        try:
            # Preprocess input
            tensor_attr = tensor_info["inputs"][0]
            s = float(tensor_attr["scale"])
            zp = int(tensor_attr["zp"])
            tensor_type = int(tensor_attr["type"])
            input_tensor, original_img, scale, pad = preprocess(image_path, new_shape=(224, 224), data_format='NHWC', s=s, zp=zp, tensor_type=tensor_type)

            # Run inference
            outputs = amlnn.inference(inputs=[input_tensor])
            print(f"outputs[0]: type={type(outputs[0])}, shape={outputs[0].shape}, dtype={outputs[0].dtype}")
            print(f"outputs[1]: type={type(outputs[1])}, shape={outputs[1].shape}, dtype={outputs[1].dtype}")

            # Postprocess results
            detections, detections_show = postprocess(outputs, scale, pad, data_format='NHWC', score_threshold=0.5, nms_threshold=0.3)
            txt_path = image_path[:image_path.rfind('.')] + "_det.txt"
            print(f"txt_path {txt_path}")
            with open(txt_path, 'w') as ofs:
                if ofs.writable():
                    for det in detections:
                        for i in range(NUM_COORDS + 1):
                            end = " " if i < NUM_COORDS else "\n"
                            ofs.write(f"{det[i]}{end}")
            print(f"Detections saved to {txt_path}")

            # Print detection results
            if detections_show:
                print(f"    Detected {len(detections_show)} objects:")
                for i, det in enumerate(detections_show, 1):
                    print(f"      {i}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("    No objects detected")

            # Save result image
            model_name = Path(args.model_path).stem
            result_dir = f"{model_name}_result"
            os.makedirs(result_dir, exist_ok=True)
            img_name = Path(image_path).stem
            save_path = os.path.join(result_dir, f"{img_name}_result.jpg")
            draw_detections(original_img, detections_show, str(save_path))
            print(f"    Result saved to: {save_path}")

        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

        print()

    print(amlnn.get_perf_info())

    # Optional visualization
    amlnn.perf_visualize()

    # Release resources
    amlnn.uninit()


if __name__ == "__main__":
    main()

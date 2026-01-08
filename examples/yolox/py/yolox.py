# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import argparse
import cv2
from pathlib import Path
from amlnnlite.api import AMLNNLite

# COCO 80 class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
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

def demo_postprocess(outputs, img_size, p6=False):
    """
    YOLOX official demo_postprocess function
    Decode model output to absolute coordinates
    """
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def preprocess(img_path, new_shape=(640, 640), data_format='NHWC'):
    """
    YOLOX preprocessing function (with ImageNet normalization)
    Returns: processed image (HWC format for NHWC, float32, normalized), scale, pad
    """
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")

    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    # Normalize to 0-1
    normalized_img = rgb_img.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_img = (normalized_img - mean) / std

    if data_format == 'NCHW':
        # HWC -> CHW -> BCHW
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        # HWC -> BHWC
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Only 'NCHW' and 'NHWC' are supported.")

    return input_tensor, original_img, scale, pad

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """
    YOLOX official multiclass_nms function (class-agnostic version)
    """
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
        return dets
    return None

def postprocess(outputs, scale, pad, img_size=(640, 640), conf_threshold=0.25, iou_threshold=0.45, p6=False):
    """
    YOLOX postprocessing (based on python_x.py)
    Assumes single output [1, 8400, 85] or multiple outputs that need to be concatenated
    """
    # Handle multiple outputs (if AMLNNLite returns multiple scales)
    if isinstance(outputs, list):
        if len(outputs) == 1:
            output = outputs[0]
        else:
            # Concatenate multiple outputs if needed
            # This assumes outputs are already in the correct format
            output = outputs[0]  # Use first output for now
    else:
        output = outputs

    # Ensure output is in correct format [1, N, 85]
    if len(output.shape) == 2:
        # [N, 85] -> [1, N, 85]
        output = output[None, :, :]
    elif len(output.shape) == 3:
        # [1, N, 85] or [N, 1, 85]
        if output.shape[0] != 1:
            output = output.transpose(1, 0, 2)[None, :, :]
    elif len(output.shape) == 4:
        # [1, 1, N, 85] -> [1, N, 85]
        output = output[0, 0]
        output = output[None, :, :]

    # Convert to float32 if needed (AMLNNLite might return int8)
    if output.dtype != np.float32:
        output = output.astype(np.float32)

    # Use demo_postprocess to decode coordinates
    predictions = demo_postprocess(output, img_size, p6=p6)[0]  # [8400, 85]

    # Extract boxes and scores
    # Format after demo_postprocess: [cx, cy, w, h, obj_conf, class0, ..., class79]
    boxes = predictions[:, :4]  # [cx, cy, w, h] (absolute coordinates)
    scores = predictions[:, 4:5] * predictions[:, 5:]  # obj_conf * cls_scores

    # Convert to xyxy format
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

    # Map coordinates back to original image
    pad_x, pad_y = pad
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale
    boxes_xyxy = np.maximum(boxes_xyxy, 0)

    # Multiclass NMS (class-agnostic, score_thr=0.1 as in official YOLOX)
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=iou_threshold, score_thr=0.1)

    if dets is None:
        return []

    # Convert to detection format
    final_boxes = dets[:, :4]
    final_scores = dets[:, 4]
    final_cls_inds = dets[:, 5].astype(int)

    detections = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = final_boxes[i]
        confidence = final_scores[i]
        class_id = final_cls_inds[i]

        if confidence >= conf_threshold:
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'class_{class_id}'
            })

    return detections

# YOLOX color palette (consistent with python_x.py)
_COLORS = (
    np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32).reshape(-1, 3)
)

def vis(img, detections, conf=0.5, class_names=None):
    """
    YOLOX official visualization function (based on python_x.py)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    result_img = img.copy()

    # Adjust font size based on image size
    img_height, img_width = img.shape[:2]
    font_scale = max(0.6, min(1.2, np.sqrt(img_height * img_height + img_width * img_width) * 0.0015))
    thickness = max(2, int(font_scale * 2.5))

    for det in detections:
        if det['confidence'] < conf:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        confidence = det['confidence']
        class_id = det['class_id']

        if class_id >= len(_COLORS):
            class_id = class_id % len(_COLORS)

        color = (_COLORS[class_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(det['class_name'], confidence * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[class_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)

        txt_bk_color = (_COLORS[class_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            result_img,
            (x1, y1 + 1),
            (x1 + txt_size[0] + 1, y1 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(result_img, text, (x1, y1 + txt_size[1]), font, font_scale, txt_color, thickness=thickness)

    return result_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='./yolox_s_int8_A311D2.adla')
    parser.add_argument('--run-cycles', default= 1, type=int)
    parser.add_argument('--input-path', default='./', help='Input image path (file or directory)')
    args = parser.parse_args()

    # Initialize AMLNNLite
    amlnn = AMLNNLite()
    amlnn.config(
        model_path=args.model_path,           # Model file path, Support ADLD and quantized TFlite models
        run_cycles=args.run_cycles
    )
    amlnn.init()

    # Find image files
    image_files = []
    if os.path.isfile(args.input_path):
        # Single image file
        image_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        # Directory - find all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.input_path, ext)))
            image_files.extend(glob.glob(os.path.join(args.input_path, ext.upper())))
    else:
        print(f"Error: Input path '{args.input_path}' does not exist")
        amlnn.uninit()
        return
    
    if not image_files:
        print(f"No image files found in {args.input_path}")
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
            input_tensor, original_img, scale, pad = preprocess(image_path, new_shape=(640, 640), data_format='NHWC')

            # Run inference
            outputs = amlnn.inference(
                inputs=[input_tensor]
            )

            # Postprocess results
            detections = postprocess(outputs, scale, pad, img_size=(640, 640), conf_threshold=0.25, iou_threshold=0.45, p6=False)

            # Print detection results
            if detections:
                print(f"    Detected {len(detections)} objects:")
                for i, det in enumerate(detections, 1):
                    print(f"      {i}. {det['class_name']} ({det['confidence']:.2f})")
            else:
                print("    No objects detected")

            # Save result image (save to current directory)
            img_name = Path(image_path).stem
            save_path = f"{img_name}_result.jpg"
            result_img = vis(original_img, detections, conf=0.25, class_names=CLASS_NAMES)
            cv2.imwrite(save_path, result_img)
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

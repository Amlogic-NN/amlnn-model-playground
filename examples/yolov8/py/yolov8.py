import numpy as np
import os
import glob
import argparse
import cv2
from pathlib import Path
from amlnnlite.api import AMLNNLite


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
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
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

def preprocess(img_path, new_shape=(640, 640), data_format='NCHW', s=0.003921568859368563, zp=-128):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")
    
    processed_img, scale, pad = letterbox(original_img, new_shape)
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
    
    # Quantize to int8
    input_tensor = np.round(input_tensor / s + zp).astype(np.int8)
    
    return input_tensor, original_img, scale, pad

def postprocess(outputs, scale, pad, data_format='NCHW', strides=[8, 16, 32], conf_threshold=0.25, iou_threshold=0.45):
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    for scale_idx, output in enumerate(outputs):
        stride = strides[scale_idx]
        
        if data_format == 'NCHW':
            # (1, 144, H, W) → (H*W, 144)
            batch_size, channels, height, width = output.shape
            output_reshaped = output.transpose(0, 2, 3, 1).reshape(-1, channels)
        elif data_format == 'NHWC':
            # (1, H, W, 144) → (H*W, 144)
            batch_size, height, width, channels = output.shape
            output_reshaped = output.reshape(-1, channels)
        else:
            raise ValueError(f"Unsupported data format: {data_format}. Only 'NCHW' and 'NHWC' are supported.")
        
        # Separate DFL and classification: 144 = 64(DFL) + 80(Classes)
        dfl_predictions = output_reshaped[:, :64]
        class_predictions = output_reshaped[:, 64:]
        
        # Apply sigmoid activation to class scores
        class_scores = 1.0 / (1.0 + np.exp(-class_predictions))
        max_class_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Generate grid coordinates
        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        grid_x = grid_x.flatten().astype(np.float32)
        grid_y = grid_y.flatten().astype(np.float32)
        
        # DFL decoding
        dfl_reshaped = dfl_predictions.reshape(-1, 4, 16)
        dfl_softmax = np.exp(dfl_reshaped) / np.sum(np.exp(dfl_reshaped), axis=-1, keepdims=True)
        regression_range = np.arange(16, dtype=np.float32)
        bbox_deltas = np.sum(dfl_softmax * regression_range[None, None, :], axis=-1)
        
        # Convert to absolute coordinates
        anchor_x = (grid_x + 0.5) * stride
        anchor_y = (grid_y + 0.5) * stride
        
        left, top, right, bottom = bbox_deltas.T
        x1 = anchor_x - left * stride
        y1 = anchor_y - top * stride
        x2 = anchor_x + right * stride
        y2 = anchor_y + bottom * stride
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        all_boxes.append(boxes)
        all_scores.append(max_class_scores)
        all_class_ids.append(class_ids)
    
    # Merge all scales
    final_boxes = np.concatenate(all_boxes, axis=0)
    final_scores = np.concatenate(all_scores, axis=0)
    final_class_ids = np.concatenate(all_class_ids, axis=0)
    
    # Filter by confidence threshold
    valid_mask = final_scores > conf_threshold
    if not np.any(valid_mask):
        return []
    
    valid_boxes = final_boxes[valid_mask]
    valid_scores = final_scores[valid_mask]
    valid_class_ids = final_class_ids[valid_mask]
    
    # Map coordinates back to original image
    pad_x, pad_y = pad
    valid_boxes[:, [0, 2]] = (valid_boxes[:, [0, 2]] - pad_x) / scale
    valid_boxes[:, [1, 3]] = (valid_boxes[:, [1, 3]] - pad_y) / scale
    valid_boxes = np.maximum(valid_boxes, 0)
    
    # NMS
    if len(valid_boxes) > 0:
        nms_indices = cv2.dnn.NMSBoxes(
            valid_boxes.tolist(), valid_scores.tolist(), conf_threshold, iou_threshold
        )
        
        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            detections = []
            
            for idx in nms_indices:
                x1, y1, x2, y2 = valid_boxes[idx]
                confidence = valid_scores[idx]
                class_id = valid_class_ids[idx]
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': class_names.get(int(class_id), f'class_{class_id}')
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
    parser.add_argument('--model-path', default='./yolov8s_int8_A311D2.adla')
    parser.add_argument('--run-cycles', default= 1, type=int)
    args = parser.parse_args()
    
    # Initialize AMLNNLite
    amlnn = AMLNNLite()
    amlnn.config(
        model_path=args.model_path,           # Model file path, Support ADLD and quantized TFlite models
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
            input_tensor, original_img, scale, pad = preprocess(image_path, new_shape=(640, 640), data_format='NHWC', s=0.003921568859368563, zp=-128)

            # Run inference
            outputs = amlnn.inference(
                inputs=[input_tensor]
            )

            # Postprocess results
            detections = postprocess(outputs, scale, pad, data_format='NHWC', strides=[8, 16, 32], conf_threshold=0.25, iou_threshold=0.45)
            
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


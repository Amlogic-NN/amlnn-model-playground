import numpy as np
import os
import glob
import argparse
import cv2
from pathlib import Path
from amlnnlite.api import AMLNNLite


class_names = [
    "handbag", "backpack", "wallet",
    "watch", "necklace", "bracelet", "earrings", "finger ring", "sunglass", "hat", "shoes", "belt",
    "makeup palette", "lipstick tube",
    "car", "truck", "bicycle", "motorcycle",
    "phone", "laptop", "camera", "wine bottle", "stuffed toy"
]

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 480
NUM_CLASSES = len(class_names)
CHANNELS = 87  # 4*16 (DFL) + 23 (classes)
STRIDES = [8, 16, 32]
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45

def letterbox(img, new_shape=(480, 640), color=(114, 114, 114)):
    """Resize and pad image with letterbox method"""
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

def preprocess(img_path, new_shape=(480, 640), data_format='NHWC'):
    """Preprocess image for YOLOWorld model (float32 input/output)"""
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        raise ValueError(f"can't read image: {img_path}")
    
    processed_img, scale, pad = letterbox(original_img, new_shape)
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype(np.float32) / 255.0

    if data_format == 'NCHW':
        # HWC -> CHW -> BCHW
        input_tensor = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
    elif data_format == 'NHWC':
        # HWC -> BHWC
        input_tensor = np.expand_dims(normalized_img, axis=0)
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Only 'NCHW' and 'NHWC' are supported.")
    
    # Keep as float32 (no quantization for float32 models)
    input_tensor = input_tensor.astype(np.float32)
    
    return input_tensor, original_img, scale, pad

def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xx1 = max(x1_1, x1_2)
    yy1 = max(y1_1, y1_2)
    xx2 = min(x2_1, x2_2)
    yy2 = min(y2_1, y2_2)
    
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    return inter / (area1 + area2 - inter + 1e-6)

def nms_by_class(detections, iou_threshold):
    """NMS within each class"""
    if len(detections) == 0:
        return []
    
    # Group by class
    class_detections = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in class_detections:
            class_detections[class_id] = []
        class_detections[class_id].append(det)
    
    final_detections = []
    for class_id, cls_dets in class_detections.items():
        # Sort by score
        cls_dets.sort(key=lambda x: x['confidence'], reverse=True)
        
        removed = [False] * len(cls_dets)
        for i in range(len(cls_dets)):
            if removed[i]:
                continue
            final_detections.append(cls_dets[i])
            
            for j in range(i + 1, len(cls_dets)):
                if removed[j]:
                    continue
                iou = compute_iou(cls_dets[i]['bbox'], cls_dets[j]['bbox'])
                if iou > iou_threshold:
                    removed[j] = True
    
    return final_detections

def suppress_cross_class_iou_conflicts(detections, iou_threshold):
    """Suppress cross-class IOU conflicts"""
    if len(detections) == 0:
        return []
    
    # Sort by score
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    removed = [False] * len(detections)
    final_detections = []
    
    for i in range(len(detections)):
        if removed[i]:
            continue
        final_detections.append(detections[i])
        
        for j in range(i + 1, len(detections)):
            if removed[j]:
                continue
            if detections[i]['class_id'] != detections[j]['class_id']:
                iou = compute_iou(detections[i]['bbox'], detections[j]['bbox'])
                if iou > iou_threshold:
                    removed[j] = True
    
    return final_detections

def get_detections(output, output_shape, stride, conf_thresh, num_classes, reverse=1, data_format='NHWC'):
    """Extract detections from a single output layer using vectorized operations"""
    coords = 4 * 16  # DFL coords: 64
    
    if data_format == 'NCHW':
        batch_size, channels, height, width = output_shape
        # Remove batch dimension and reshape: (channels, height, width) -> (height * width, channels)
        output_reshaped = output[0].transpose(1, 2, 0).reshape(-1, channels)
    elif data_format == 'NHWC':
        batch_size, height, width, channels = output_shape
        # Remove batch dimension and reshape: (height, width, channels) -> (height * width, channels)
        output_reshaped = output[0].reshape(-1, channels)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    # reverse=0: standard YOLO [classes + box]
    # reverse>0: YOLOWorld [box + classes]
    cls_offset = coords if reverse > 0 else 0
    dfl_offset = 0 if reverse > 0 else num_classes
    
    # Extract class predictions and apply sigmoid
    class_predictions = output_reshaped[:, cls_offset:cls_offset + num_classes]
    class_scores = sigmoid(class_predictions)
    
    # Get max class scores and class IDs
    max_class_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Filter by confidence threshold
    valid_mask = max_class_scores > conf_thresh
    if not np.any(valid_mask):
        return []
    
    # Extract DFL predictions for valid detections
    dfl_predictions = output_reshaped[valid_mask, dfl_offset:dfl_offset + coords]
    valid_scores = max_class_scores[valid_mask]
    valid_class_ids = class_ids[valid_mask]
    
    # Reshape DFL: (N, 64) -> (N, 4, 16)
    dfl_reshaped = dfl_predictions.reshape(-1, 4, 16)
    
    # DFL decoding with softmax
    max_logits = np.max(dfl_reshaped, axis=-1, keepdims=True)
    dfl_exp = np.exp(dfl_reshaped - max_logits)
    dfl_softmax = dfl_exp / np.sum(dfl_exp, axis=-1, keepdims=True)
    
    # Weighted sum: regression_range = [0, 1, 2, ..., 15]
    regression_range = np.arange(16, dtype=np.float32)
    bbox_deltas = np.sum(dfl_softmax * regression_range[None, :], axis=-1)  # (N, 4)
    
    # Generate grid coordinates
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    
    # Filter grid coordinates
    valid_grid_x = grid_x[valid_mask]
    valid_grid_y = grid_y[valid_mask]
    
    # Convert to absolute coordinates
    anchor_x = (valid_grid_x + 0.5) * stride
    anchor_y = (valid_grid_y + 0.5) * stride
    
    left, top, right, bottom = bbox_deltas.T
    x1 = anchor_x - left * stride
    y1 = anchor_y - top * stride
    x2 = anchor_x + right * stride
    y2 = anchor_y + bottom * stride
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Create detections list
    detections = []
    for i in range(len(boxes)):
        detections.append({
            'bbox': [float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3])],
            'confidence': float(valid_scores[i]),
            'class_id': int(valid_class_ids[i])
        })
    
    return detections

def postprocess(outputs, scale, pad, data_format='NHWC', strides=[8, 16, 32], 
                conf_threshold=0.4, iou_threshold=0.45, num_classes=23, reverse=1):
    """Postprocess YOLOWorld outputs"""
    all_detections = []
    
    # Process each output scale
    for scale_idx, output in enumerate(outputs):
        stride = strides[scale_idx]
        
        # Output should already be float32 (no dequantization needed)
        if output.dtype != np.float32:
            output = output.astype(np.float32)
        
        if data_format == 'NCHW':
            batch_size, channels, height, width = output.shape
            output_shape = (batch_size, channels, height, width)
        elif data_format == 'NHWC':
            batch_size, height, width, channels = output.shape
            output_shape = (batch_size, height, width, channels)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        dets = get_detections(output, output_shape, stride, conf_threshold, 
                             num_classes, reverse, data_format)
        all_detections.extend(dets)
    
    if len(all_detections) == 0:
        return []
    
    # Map coordinates back to original image
    pad_x, pad_y = pad
    detections_orig = []
    for det in all_detections:
        x1, y1, x2, y2 = det['bbox']
        x1_orig = (x1 - pad_x) / scale
        y1_orig = (y1 - pad_y) / scale
        x2_orig = (x2 - pad_x) / scale
        y2_orig = (y2 - pad_y) / scale
        
        detections_orig.append({
            'bbox': [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)],
            'confidence': det['confidence'],
            'class_id': det['class_id'],
            'class_name': class_names[det['class_id']] if det['class_id'] < len(class_names) else f'class_{det["class_id"]}'
        })
    
    # NMS by class
    detections_nms = nms_by_class(detections_orig, iou_threshold)
    
    # Suppress cross-class IOU conflicts
    final_detections = suppress_cross_class_iou_conflicts(detections_nms, 0.8)
    
    return final_detections

def get_class_color(class_id):
    """Generate a color for each class"""
    import colorsys
    hue = (class_id * 137.508) % 360
    rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
    bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    return bgr

def draw_detections(img, detections, save_path):
    """Draw detection results on image"""
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
    parser = argparse.ArgumentParser(
        description='YOLOWorld object detection demo using AMLNNLite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default model path
  python yoloworld.py
  
  # Specify model path
  python yoloworld.py --model-path ./model.adla
  
  # Run multiple cycles for performance testing
  python yoloworld.py --run-cycles 100
        '''
    )
    parser.add_argument('--model-path', 
                       default='./yolo_world_480_640.adla',
                       help='Path to the model file (.adla or .tflite)')
    parser.add_argument('--run-cycles', 
                       default=1, 
                       type=int,
                       help='Number of inference cycles to run (for performance testing)')
    parser.add_argument('--image-dir',
                       default='./',
                       help='Directory containing images to process')
    parser.add_argument('--conf-threshold',
                       type=float,
                       default=SCORE_THRESHOLD,
                       help=f'Confidence threshold for detection (default: {SCORE_THRESHOLD})')
    parser.add_argument('--nms-threshold',
                       type=float,
                       default=NMS_THRESHOLD,
                       help=f'NMS IoU threshold (default: {NMS_THRESHOLD})')
    parser.add_argument('--no-visualize',
                       action='store_true',
                       help='Skip visualization chart generation')
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    if not os.path.isfile(args.model_path):
        print(f"Error: Model path is not a file: {args.model_path}")
        return 1
    
    # Validate thresholds
    if not 0.0 < args.conf_threshold <= 1.0:
        print(f"Error: Confidence threshold must be in (0, 1], got {args.conf_threshold}")
        return 1
    
    if not 0.0 < args.nms_threshold <= 1.0:
        print(f"Error: NMS threshold must be in (0, 1], got {args.nms_threshold}")
        return 1
    
    # Initialize AMLNNLite with error handling
    print("Initializing AMLNNLite...")
    amlnn = None
    try:
        amlnn = AMLNNLite()
        print(f"Loading model: {args.model_path}")
        amlnn.config(
            model_path=args.model_path,
            run_cycles=args.run_cycles
        )
        print("Initializing model...")
        amlnn.init()
        print("Model initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing AMLNNLite: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Use try-finally to ensure resources are released
    try:
        # Find all image files in the specified directory
        image_dir = args.image_dir
        if not os.path.exists(image_dir):
            print(f"Error: Image directory not found: {image_dir}")
            return 1
        
        if not os.path.isdir(image_dir):
            print(f"Error: Image path is not a directory: {image_dir}")
            return 1
        
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
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
                # Preprocess input (float32 model, no quantization)
                input_tensor, original_img, scale, pad = preprocess(
                    image_path, 
                    new_shape=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), 
                    data_format='NHWC'
                )
                
                # Validate input tensor shape and dtype
                expected_shape = (1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3)
                if input_tensor.shape != expected_shape:
                    raise ValueError(f"Input tensor shape mismatch: expected {expected_shape}, got {input_tensor.shape}")
                if input_tensor.dtype != np.float32:
                    raise ValueError(f"Input tensor dtype must be float32, got {input_tensor.dtype}")

                # Run inference
                outputs = amlnn.inference(inputs=[input_tensor])
                
                # Validate outputs
                if outputs is None:
                    raise ValueError("Inference returned None")
                if len(outputs) != 3:
                    raise ValueError(f"Expected 3 output tensors, got {len(outputs)}")
                
                # Postprocess results
                detections = postprocess(
                    outputs, 
                    scale, 
                    pad, 
                    data_format='NHWC', 
                    strides=STRIDES, 
                    conf_threshold=args.conf_threshold, 
                    iou_threshold=args.nms_threshold,
                    num_classes=NUM_CLASSES,
                    reverse=1  # YOLOWorld format
                )
                
                # Print detection results
                if detections:
                    print(f"    Detected {len(detections)} object(s):")
                    for j, det in enumerate(detections, 1):
                        bbox = det['bbox']
                        print(f"{j}. {det['class_name']} ({det['confidence']:.2f})")
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
                import traceback
                traceback.print_exc()
                # Continue processing other images
                continue
            
            print()

        # Optional visualization
        if not args.no_visualize:
            print("Generating visualization charts...")
            amlnn.visualize()
            print("Visualization charts saved.")
    finally:
        # Always release resources
        if amlnn is not None:
            print("\nReleasing resources...")
            amlnn.uninit()
            print("Resources released.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())



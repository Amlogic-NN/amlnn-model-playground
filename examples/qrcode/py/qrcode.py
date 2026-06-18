# -*- coding: utf-8 -*-
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

import os
import cv2
import glob
import argparse
import numpy as np
from pathlib import Path
from amlnn.api import AMLNN

MEAN = np.array([0, 0, 0], dtype=np.float32)
STD  = np.array([255, 255, 255], dtype=np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms_xyxy(boxes, scores, iou_thres=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def preprocess(img_path, input_size=(320, 320), tensor_type=0):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None

    orig = img.copy()
    img = cv2.resize(img, input_size)

    if tensor_type == 0:
        img = img.astype(np.float32) / 255.0
    elif tensor_type == 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float16)

    img = np.expand_dims(img, axis=0)
    # img = (img / 0.003776 - 128).astype(np.int8)
    return img, orig, orig.shape[:2]


def postprocess_qrcode(outputs, orig_shape, conf_thres=0.8, nms_thres=0.5, pad=40):
    h0, w0 = orig_shape
    sx = w0 / 320.0
    sy = h0 / 320.0

    out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    out = np.asarray(out)

    if out.ndim == 4 and out.shape[0] == 1 and out.shape[1] == 1:
        out = out[0, 0]
        pred = out.transpose(1, 0)
    elif out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
        pred = out.transpose(1, 0)
    else:
        raise ValueError(f"Unexpected output shape: {out.shape}")

    boxes_xyxy_320 = []
    scores = []

    for i in range(pred.shape[0]):
        x, y, w, h, raw_score = pred[i]

        score = sigmoid(raw_score)

        if score < conf_thres:
            continue

        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0

        boxes_xyxy_320.append([float(x1), float(y1), float(x2), float(y2)])
        scores.append(float(score))

    keep = nms_xyxy(boxes_xyxy_320, scores, iou_thres=nms_thres)

    results = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_xyxy_320[idx]
        score = scores[idx]

        x1 = x1 * sx
        y1 = y1 * sy
        x2 = x2 * sx
        y2 = y2 * sy

        x1p = int(max(0, x1 - pad-10))
        y1p = int(max(0, y1 - pad-15))
        x2p = int(min(w0 - 1, x2 + pad-10))
        y2p = int(min(h0 - 1, y2 + pad))

        results.append({
            "box": [x1p, y1p, x2p, y2p],
            "score": score,
        })

    return results


def decode_qrcodes(orig, dets):
    detector = cv2.QRCodeDetector()
    decoded = []

    h, w = orig.shape[:2]

    for det in dets:
        x1, y1, x2, y2 = map(int, det["box"])
        score = det["score"]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            print(f"skip invalid box: {[x1, y1, x2, y2]}, score={score:.3f}")
            continue

        crop = orig[y1:y2, x1:x2].copy()
        if crop.size == 0:
            print(f"skip empty crop: {[x1, y1, x2, y2]}, score={score:.3f}")
            continue

        ch, cw = crop.shape[:2]
        if cw < 20 or ch < 20:
            print(f"skip too small crop: {[x1, y1, x2, y2]}, size=({cw},{ch}), score={score:.3f}")
            continue

        try:
            text, points, _ = detector.detectAndDecode(crop)
        except cv2.error as e:
            print(f"skip cv2 decode error: {[x1, y1, x2, y2]}, score={score:.3f}, err={e}")
            continue

        decoded.append({
            "box": [x1, y1, x2, y2],
            "score": score,
            "text": text,
            "points": points,
        })

    return decoded


def draw_results(img, results):
    vis = img.copy()
    for r in results:
        x1, y1, x2, y2 = r["box"]
        score = r["score"]
        text = r["text"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis, f"{score:.3f}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        if text:
            cv2.putText(
                vis, text[:60], (x1, min(img.shape[0] - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )
    return vis


def main():
    parser = argparse.ArgumentParser(description="QRCode AMLNN Demo")
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--dataset-path', required=True, help='Path to quant dataset')
    parser.add_argument('--image-dir', required=True, help='Directory of test images')
    parser.add_argument('--conf-thres', type=float, default=0.8)
    parser.add_argument('--nms-thres', type=float, default=0.5)
    parser.add_argument('--pad', type=int, default=40)
    parser.add_argument('--target-platform', required=True, help='Platform ID, e.g. 001, 002, 003')
    args = parser.parse_args()

    amlnn = AMLNN()

    amlnn.load_tflite(model=args.model_path, quantized_model=True)
    # amlnn.load_pytorch(model=args.model_path, input_shapes=[[1,3,320,320]], dtypes=["float32"])
    # amlnn.load_onnx(model=args.model_path)

    amlnn.config(normalization_mean=[MEAN.tolist()], normalization_std=[STD.tolist()], quantized_dtype='w8a8', target_platform=f"PRODUCT_PID0XA{args.target_platform.zfill(3)}")

    amlnn.compile(dataset=args.dataset_path)

    amlnn.export_adla()

    amlnn.init_runtime(mode="native", enable_perf=True)

    tensor_info = amlnn.get_tensor_info()

    print(amlnn.get_sdk_version())


    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.[jp][pn][g]")))
    if not image_files:
        print(f"No images found in {args.image_dir}")
        amlnn.uninit()
        return

    res_dir = "qrcode_result"
    os.makedirs(res_dir, exist_ok=True)

    for idx, img_path in enumerate(image_files, start=1):
        print("=" * 60)
        print(f"Processing image {idx}/{len(image_files)}: {Path(img_path).name}")
        print("=" * 60)

        tensor_attr = tensor_info["inputs"][0]
        tensor_type = int(tensor_attr["type"])
        inp, orig, orig_shape = preprocess(img_path, tensor_type=tensor_type)
        if inp is None:
            print(f"Failed to read: {img_path}")
            continue

        outputs = amlnn.inference(inp)
        dets = postprocess_qrcode(
            outputs,
            orig_shape,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            pad=args.pad
        )
        results = decode_qrcodes(orig, dets)

        if len(results) == 0:
            print("    No objects detected")
        else:
            print(f"    Detected {len(results)} objects:")
            for i, r in enumerate(results, 1):
                print(f"      {i}. score={r['score']:.3f}")
                print(f"         box={r['box']}")
                print(f"         text={r['text']}")

        vis = draw_results(orig, results)
        save_path = os.path.join(res_dir, Path(img_path).name)
        cv2.imwrite(save_path, vis)
        print(f"    Result saved to: {save_path}")

    print(amlnn.get_perf_info())

    amlnn.perf_visualize()

    amlnn.uninit()


if __name__ == "__main__":
    main()
    
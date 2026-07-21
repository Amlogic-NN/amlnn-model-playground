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
"""
MobileSAM 448 ADLA (Amlogic NPU) standalone inference (no web UI).

Port of the 448 TFLite CLI to ADLA. Runs encoder + decoder on the NPU for a single
image and a point or box prompt, then saves an overlay image and a binary mask PNG.

Usage — single point (p2):
  python infer_mobilesam_adla_448.py \
    --image ./picture.jpg \
    --encoder ./models/448/mobile_sam_encoder_448_w8a16.adla \
    --decoder ./models/448/mobile_sam_decoder_448_no_post_p2.adla \
    --points "500,1000,1" \
    --output ./result_adla_448_point.png

Usage — box (two corners, no padding):
  python infer_mobilesam_adla_448.py \
    --image ./picture.jpg \
    --encoder ./models/448/mobile_sam_encoder_448_w8a16.adla \
    --decoder ./models/448/mobile_sam_decoder_448_no_post_p2.adla \
    --box "450,750,800,1000" \
    --output ./result_adla_448_box.png

Notes:
  - AMLNN only accepts 1D or 4D inputs, so point_coords [1,N,2] -> [1,1,N,2] and
    point_labels [1,N] -> [1,1,1,N] before being fed in.
  - Inputs are matched to the model's slots by element count; sizes (input, embeddings,
    mask_input, low-res masks) are read from the model, so this also works for other
    resolutions.
  - AMLNN perf is intentionally NOT enabled (it can break the invoke on some firmware).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from amlnn.api import AMLNN


SAM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Labels for the two box corners. The 448 TFLite reference used two positive points
# (label 1) with --no-padding, so that is the default here. If your model expects the
# standard SAM box encoding instead, set this to (2.0, 3.0).
BOX_LABELS = (2.0, 3.0)

TYPE_TO_NP = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32, 4: np.uint8, 5: np.int64}
NAME_TO_NP = {
    "FP32": np.float32, "FLOAT32": np.float32, "FP16": np.float16, "FLOAT16": np.float16,
    "INT8": np.int8, "UINT8": np.uint8, "INT32": np.int32, "INT64": np.int64,
}


# ================================================================ AMLNN wrapper
def _dims(attr):
    for k in ("dims", "shape", "dim", "size"):
        v = attr.get(k) if isinstance(attr, dict) else None
        if v is None:
            continue
        try:
            return tuple(int(x) for x in v)
        except (TypeError, ValueError):
            continue
    return None


def _numel(attr):
    d = _dims(attr)
    return int(np.prod(d)) if d else None


class AdlaModel:
    """
    Thin AMLNN wrapper. infer() takes {logical_name: ndarray} (each already in the
    model's required 4D/1D shape), orders them by the model's input order (matched by
    element count, unique for these models), casts dtypes, then calls inference.
    """

    def __init__(self, path, tag):
        self.tag = tag
        self.net = AMLNN()
        self.net.init_runtime(mode="native")   # perf stays off on purpose
        self.net.load_model(path=str(path))
        info = self.net.get_tensor_info()
        self.in_attrs = info.get("inputs", [])
        self.out_attrs = info.get("outputs", [])
        self._print_info()

    def _print_info(self):
        print(f"[{self.tag}] inputs:")
        for i, a in enumerate(self.in_attrs):
            print(f"    in [{i}] name={a.get('name','')} dims={_dims(a)} "
                  f"type={a.get('type_name', a.get('type'))}")
        print(f"[{self.tag}] outputs:")
        for i, a in enumerate(self.out_attrs):
            print(f"    out[{i}] name={a.get('name','')} dims={_dims(a)} "
                  f"type={a.get('type_name', a.get('type'))}")

    def _in_dtype(self, i):
        a = self.in_attrs[i]
        name = str(a.get("type_name", "")).strip().upper()
        if name in NAME_TO_NP:
            return NAME_TO_NP[name]
        try:
            return TYPE_TO_NP.get(int(a.get("type", 0)), np.float32)
        except (TypeError, ValueError):
            return np.float32

    def _order(self, named, fallback_order):
        model_numels = [_numel(a) for a in self.in_attrs]
        if len(model_numels) == len(named) and all(n is not None for n in model_numels):
            pool = {name: int(np.prod(arr.shape)) for name, arr in named.items()}
            ordered, remaining = [], dict(named)
            ok = True
            for mn in model_numels:
                match = next((nm for nm in remaining if pool[nm] == mn), None)
                if match is None:
                    ok = False
                    break
                ordered.append(remaining.pop(match))
            if ok and not remaining:
                return ordered
        return [named[name] for name in fallback_order]

    def infer(self, named, fallback_order):
        arrays = self._order(named, fallback_order)
        casted = [
            np.ascontiguousarray(arr).astype(self._in_dtype(i))
            for i, arr in enumerate(arrays)
        ]
        outs = self.net.inference(
            inputs=casted,
            inputs_data_format="NHWC",
            outputs_data_format="NHWC",
        )
        if not outs:
            raise RuntimeError(f"[{self.tag}] inference returned empty outputs")
        return outs

    def uninit(self):
        try:
            self.net.uninit()
        except Exception:
            pass


# ================================================================ pre/post-processing
def preprocess_image(image_rgb, target_size):
    """Long-side resize to target_size, then pad to target_size x target_size."""
    orig_h, orig_w = image_rgb.shape[:2]
    scale = target_size / max(orig_h, orig_w)
    new_h = int(orig_h * scale + 0.5)
    new_w = int(orig_w * scale + 0.5)

    resized = np.array(
        Image.fromarray(image_rgb).resize((new_w, new_h), resample=Image.BILINEAR)
    ).astype(np.float32)

    normalized = (resized - SAM_MEAN) / SAM_STD
    padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = normalized

    meta = {
        "orig_h": orig_h, "orig_w": orig_w,
        "new_h": new_h, "new_w": new_w,
        "x_scale": new_w / orig_w, "y_scale": new_h / orig_h,
        "input_size": target_size,
    }
    return padded[None].astype(np.float32), meta


def parse_points(points_str):
    """Format: "x,y,label" or "x1,y1,l1;x2,y2,l2". label: 1=positive, 0=negative."""
    points = []
    for item in points_str.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid point format: {item}. Expected x,y,label")
        x, y, label = float(parts[0]), float(parts[1]), int(parts[2])
        if label not in (0, 1):
            raise ValueError(f"Point label must be 0 or 1, got {label}")
        points.append((x, y, label))
    if not points:
        raise ValueError("No valid points found.")
    return points


def build_prompt(points, num_points, meta, add_padding):
    """
    points: list of (x, y, label) in ORIGINAL image coordinates.
    Returns point_coords [1,num_points,2], point_labels [1,num_points]
    (coordinates scaled into the resized/input space; padding slots = -1).
    """
    required = len(points) + (1 if add_padding else 0)
    if required > num_points:
        raise ValueError(
            f"Got {len(points)} real points, add_padding={add_padding}, "
            f"required={required}, but decoder num_points={num_points}"
        )

    coords = np.zeros((1, num_points, 2), dtype=np.float32)
    labels = np.full((1, num_points), -1, dtype=np.float32)
    for i, (x, y, label) in enumerate(points):
        coords[0, i, 0] = float(x) * meta["x_scale"]
        coords[0, i, 1] = float(y) * meta["y_scale"]
        labels[0, i] = float(label)
    return coords, labels


def postprocess_masks_nhwc(low_res_masks, meta):
    """[1,Hlr,Wlr,C] -> low-res -> input_size -> crop padding -> original size."""
    orig_h, orig_w = meta["orig_h"], meta["orig_w"]
    new_h, new_w = meta["new_h"], meta["new_w"]
    input_size = meta["input_size"]

    masks = low_res_masks[0]
    processed = []
    for i in range(masks.shape[-1]):
        m = masks[:, :, i]
        m_input = cv2.resize(m, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        m_crop = m_input[:new_h, :new_w]
        m_orig = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        processed.append(m_orig)
    return np.stack(processed, axis=-1)[None].astype(np.float32)


def overlay(image_rgb, mask, points=None, box=None, alpha=0.55):
    out = image_rgb.copy()
    color = np.array([30, 255, 30], dtype=np.uint8)
    ov = out.copy()
    ov[mask] = color
    out = cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)

    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (88, 166, 255), 2)

    for x, y, label in (points or []):
        x, y = int(round(x)), int(round(y))
        pc = (255, 0, 0) if label == 1 else (0, 0, 255)  # red=positive, blue=negative
        cv2.circle(out, (x, y), 9, (255, 255, 255), -1)
        cv2.circle(out, (x, y), 6, pc, -1)
        cv2.circle(out, (x, y), 9, (0, 0, 0), 2)
    return out


# ================================================================ shape discovery
def discover(encoder, decoder):
    """Read IMG_SIZE, embeddings shape, mask_input shape, mask-output shape from models."""
    enc_in = _dims(encoder.in_attrs[0])
    img_size = int(enc_in[1]) if enc_in and len(enc_in) == 4 else 448
    emb_shape = _dims(encoder.out_attrs[0])
    emb_numel = int(np.prod(emb_shape))

    # mask_input = the largest decoder input that is not the embeddings
    non_emb = [(_dims(a), _numel(a)) for a in decoder.in_attrs if _numel(a) != emb_numel]
    mask_input_shape = max(non_emb, key=lambda t: t[1])[0]

    # mask output = the largest decoder output
    outs = [(_dims(a), _numel(a)) for a in decoder.out_attrs]
    mask_out_shape = max(outs, key=lambda t: t[1])[0]

    return img_size, emb_shape, mask_input_shape, mask_out_shape


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--encoder", required=True, help="448 encoder .adla")
    ap.add_argument("--decoder", required=True, help="448 decoder .adla (p2)")
    ap.add_argument("--points", default=None,
                    help='Point prompt: "x,y,label" or "x1,y1,l1;x2,y2,l2" (label 1=pos,0=neg)')
    ap.add_argument("--box", default=None,
                    help='Box prompt: "x1,y1,x2,y2" (two corners, no padding)')
    ap.add_argument("--decoder-num-points", type=int, default=2,
                    help="Decoder fixed num_points (p2 -> 2)")
    ap.add_argument("--no-padding", action="store_true",
                    help="Do not append the -1 padding point (used for 2-point / box prompts)")
    ap.add_argument("--mask-index", type=int, default=-1,
                    help="Force mask index; default -1 = argmax score")
    ap.add_argument("--output", default="./result_adla_448_overlay.png")
    args = ap.parse_args()

    if not args.points and not args.box:
        ap.error("Provide either --points or --box")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print("Input image:", image_rgb.shape)

    print("Loading encoder:", args.encoder)
    encoder = AdlaModel(args.encoder, tag="encoder")
    print("Loading decoder:", args.decoder)
    decoder = AdlaModel(args.decoder, tag="decoder")

    img_size, emb_shape, mask_input_shape, mask_out_shape = discover(encoder, decoder)
    num_points = args.decoder_num_points
    print(f"IMG_SIZE={img_size} emb={emb_shape} mask_input={mask_input_shape} "
          f"mask_out={mask_out_shape} num_points={num_points}")

    input_image, meta = preprocess_image(image_rgb, img_size)

    # ---- build the prompt (point or box), in canonical shapes ----
    draw_points = None
    draw_box = None
    if args.box:
        vals = [float(v) for v in args.box.split(",")]
        if len(vals) != 4:
            raise ValueError('--box must be "x1,y1,x2,y2"')
        x1, x2 = sorted(vals[0:3:2])
        y1, y2 = sorted(vals[1:4:2])
        # two corner points, no padding
        pts = [(x1, y1, BOX_LABELS[0]), (x2, y2, BOX_LABELS[1])]
        coords, labels = build_prompt(pts, num_points, meta, add_padding=False)
        draw_box = (x1, y1, x2, y2)
        print("Box prompt:", draw_box, "labels:", BOX_LABELS)
    else:
        pts = parse_points(args.points)
        coords, labels = build_prompt(
            pts, num_points, meta, add_padding=not args.no_padding
        )
        draw_points = pts
        print("Point prompt:", pts, "add_padding:", not args.no_padding)

    # ---- encoder ----
    print("Running encoder...")
    enc_out = encoder.infer(
        named={"input_image": input_image}, fallback_order=["input_image"]
    )
    emb = np.asarray(enc_out[0], dtype=np.float32).reshape(emb_shape)
    print("image_embeddings:", emb.shape)

    # ---- decoder (point tensors expanded to 4D for AMLNN) ----
    print("Running decoder...")
    named = {
        "image_embeddings": emb,
        "point_coords": coords.reshape(1, 1, num_points, 2),
        "point_labels": labels.reshape(1, 1, 1, num_points),
        "mask_input": np.zeros(mask_input_shape, dtype=np.float32),
        "has_mask_input": np.zeros((1,), dtype=np.float32),
    }
    fallback = ["image_embeddings", "point_coords", "point_labels",
                "mask_input", "has_mask_input"]
    dec_out = decoder.infer(named=named, fallback_order=fallback)

    arrs = [np.asarray(o, dtype=np.float32) for o in dec_out]
    arrs.sort(key=lambda a: a.size, reverse=True)
    low_res_masks = arrs[0].reshape(mask_out_shape)
    scores = arrs[1].reshape(1, -1)
    print("low_res_masks:", low_res_masks.shape, "scores:", scores.ravel())

    # ---- postprocess ----
    upscaled = postprocess_masks_nhwc(low_res_masks, meta)
    num_masks = upscaled.shape[-1]
    if args.mask_index >= 0:
        if args.mask_index >= num_masks:
            raise ValueError(f"--mask-index {args.mask_index} out of range (num_masks={num_masks})")
        best_idx = args.mask_index
    else:
        best_idx = int(np.argmax(scores[0]))
    print("best_idx:", best_idx, "best_score:", float(scores[0, best_idx]))

    mask = upscaled[0, :, :, best_idx] > 0.0

    output_path = Path(args.output)
    mask_path = output_path.with_suffix(".mask.png")
    cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)

    vis_rgb = overlay(image_rgb, mask, points=draw_points, box=draw_box)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

    encoder.uninit()
    decoder.uninit()

    print(f"Saved overlay: {output_path}")
    print(f"Saved mask:    {mask_path}")


if __name__ == "__main__":
    main()

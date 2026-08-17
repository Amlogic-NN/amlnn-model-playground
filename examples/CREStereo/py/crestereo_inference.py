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
import argparse
import cv2
from pathlib import Path
from amlnn.api import AMLNN


def preprocess(img, new_shape, s, zp, tensor_type):
    input_h, input_w = new_shape

    # 1. Resize directly to model input resolution
    processed_img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_AREA)

    # 2. BGR to RGB
    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    rgb_float = rgb_img.astype(np.float32)

    # 3. No normalization. Quantize only when the model input requires it.
    if tensor_type == 0: # FP32 & FP16 host input
        input_tensor = rgb_float
    elif tensor_type in (2, 3, 4):
        inv_scale = np.float32(1.0 / s)
        raw_val = np.round(rgb_float * inv_scale + zp)

        if tensor_type == 2:    # Int8
            input_tensor = np.clip(raw_val, -128, 127).astype(np.int8)
        elif tensor_type == 3:  # Uint8
            input_tensor = np.clip(raw_val, 0, 255).astype(np.uint8)
        else:                   # Int16
            input_tensor = np.clip(raw_val, -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Does not support tensor type: {tensor_type}")

    input_tensor = np.expand_dims(input_tensor, axis=0)
    return np.ascontiguousarray(input_tensor)


def postprocess(outputs, output_shape, original_shape):
    if len(outputs) != 1:
        raise RuntimeError(f"Expected 1 CREStereo output, got {len(outputs)}")

    raw_output = np.asarray(outputs[0], dtype=np.float32).reshape(output_shape)
    if raw_output.ndim != 4 or raw_output.shape[0] != 1 or raw_output.shape[-1] < 1:
        raise RuntimeError(f"Expected output shape [1, H, W, C], got {raw_output.shape}")

    # Output channel 0 is horizontal disparity.
    disparity = raw_output[0, :, :, 0].astype(np.float32, copy=True)
    disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

    original_h, original_w = original_shape
    model_h, model_w = disparity.shape
    disparity = cv2.resize(disparity, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    disparity *= float(original_w) / float(model_w)

    return disparity


def colorize_disparity(disparity):
    finite_mask = np.isfinite(disparity)
    if not np.any(finite_mask):
        return cv2.applyColorMap(np.zeros(disparity.shape, np.uint8), cv2.COLORMAP_MAGMA)

    valid = disparity[finite_mask]
    min_value = float(np.min(valid))
    max_value = float(np.max(valid))

    if max_value <= min_value:
        normalized = np.zeros(disparity.shape, dtype=np.uint8)
    else:
        normalized_float = 255.0 * (disparity - min_value) / (max_value - min_value)
        normalized_float[~finite_mask] = 0.0
        normalized = cv2.convertScaleAbs(normalized_float, alpha=1.0)

    return cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)


def main():
    parser = argparse.ArgumentParser(description="CREStereo Demo")
    parser.add_argument('--adla', required=True, help='Path to .adla model')
    parser.add_argument('--left', required=True, help='Left image')
    parser.add_argument('--right', required=True, help='Right image')
    args = parser.parse_args()

    amlnn = AMLNN()
    amlnn.init_runtime(mode="native")

    try:
        amlnn.load_model(path=args.adla)
        tensor_info = amlnn.get_tensor_info()

        if len(tensor_info["inputs"]) != 4:
            raise RuntimeError(f"Expected 4 CREStereo inputs, got {len(tensor_info['inputs'])}")
        if len(tensor_info["outputs"]) != 1:
            raise RuntimeError(f"Expected 1 CREStereo output, got {len(tensor_info['outputs'])}")

        print(amlnn.get_sdk_version())

        left_img = cv2.imread(str(args.left))
        if left_img is None:
            raise ValueError(f"can't read image: {args.left}")

        right_img = cv2.imread(str(args.right))
        if right_img is None:
            raise ValueError(f"can't read image: {args.right}")

        if left_img.shape[:2] != right_img.shape[:2]:
            raise ValueError(f"left/right sizes differ: left={left_img.shape[:2]}, right={right_img.shape[:2]}")

        original_h, original_w = left_img.shape[:2]

        init_left_attr = tensor_info["inputs"][0]
        init_right_attr = tensor_info["inputs"][1]
        next_left_attr = tensor_info["inputs"][2]
        next_right_attr = tensor_info["inputs"][3]

        init_left_dims = tuple(int(dim) for dim in init_left_attr["dims"])
        init_right_dims = tuple(int(dim) for dim in init_right_attr["dims"])
        next_left_dims = tuple(int(dim) for dim in next_left_attr["dims"])
        next_right_dims = tuple(int(dim) for dim in next_right_attr["dims"])

        if init_left_dims != init_right_dims:
            raise RuntimeError(f"Init left/right input shapes differ: {init_left_dims} vs {init_right_dims}")
        if next_left_dims != next_right_dims:
            raise RuntimeError(f"Next left/right input shapes differ: {next_left_dims} vs {next_right_dims}")
        if len(init_left_dims) != 4 or init_left_dims[0] != 1 or init_left_dims[3] != 3:
            raise RuntimeError(f"Unexpected init input shape: {init_left_dims}")
        if len(next_left_dims) != 4 or next_left_dims[0] != 1 or next_left_dims[3] != 3:
            raise RuntimeError(f"Unexpected next input shape: {next_left_dims}")

        init_shape = (init_left_dims[1], init_left_dims[2])
        input_shape = (next_left_dims[1], next_left_dims[2])

        init_left = preprocess(left_img, init_shape, float(init_left_attr["scale"]), int(init_left_attr["zp"]), int(init_left_attr["type"]))
        init_right = preprocess(right_img, init_shape, float(init_right_attr["scale"]), int(init_right_attr["zp"]), int(init_right_attr["type"]))
        next_left = preprocess(left_img, input_shape, float(next_left_attr["scale"]), int(next_left_attr["zp"]), int(next_left_attr["type"]))
        next_right = preprocess(right_img, input_shape, float(next_right_attr["scale"]), int(next_right_attr["zp"]), int(next_right_attr["type"]))

        # Run inference
        outputs = amlnn.inference(inputs=[init_left, init_right, next_left, next_right])

        # Postprocess results
        output_shape = tuple(int(dim) for dim in tensor_info["outputs"][0]["dims"])
        disparity = postprocess(outputs, output_shape, (original_h, original_w))

        # Save result image
        result_dir = f"{Path(args.adla).stem}_result"
        os.makedirs(result_dir, exist_ok=True)
        save_path = os.path.join(result_dir, f"{Path(args.left).stem}_result.jpg")
        color_disparity = colorize_disparity(disparity)
        if not cv2.imwrite(save_path, color_disparity):
            raise RuntimeError(f"Failed to save result image: {save_path}")
        print(f"    Result saved to: {save_path}")
    finally:
        amlnn.uninit()


if __name__ == "__main__":
    main()
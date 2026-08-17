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
from amlnn.api import AMLNN

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def prepare_input_tensor(input_float, s, zp, tensor_type):
    if tensor_type == 0:
        return input_float.astype(np.float32)

    raw_val = np.round(input_float / s + zp)

    if tensor_type == 2:
        return np.clip(raw_val, -128, 127).astype(np.int8)
    elif tensor_type == 3:
        return np.clip(raw_val, 0, 255).astype(np.uint8)
    elif tensor_type == 4:
        return np.clip(raw_val, -32768, 32767).astype(np.int16)

    raise ValueError(f"Does not support tensor type: {tensor_type}")


def preprocess(image_path, new_shape, s, zp, tensor_type):
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        raise ValueError(f"can't read image: {image_path}")

    input_height, input_width = new_shape
    resized_img = cv2.resize(
        original_img,
        (input_width, input_height),
        interpolation=cv2.INTER_CUBIC
    )

    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    input_float = rgb_img.astype(np.float32)
    input_float = (input_float - IMAGENET_MEAN) / IMAGENET_STD
    input_float = np.expand_dims(input_float, axis=0)

    input_tensor = prepare_input_tensor(
        input_float,
        s,
        zp,
        tensor_type
    )

    return input_tensor, original_img


def reshape_for_model_input(input_float, input_dims):
    expected_elements = int(np.prod(input_dims))

    if input_float.size != expected_elements:
        raise RuntimeError(
            f"Depth input shape {tuple(input_dims)} contains "
            f"{expected_elements} elements, but concatenated backbone "
            f"features contain {input_float.size}"
        )

    return input_float.reshape(
        tuple(int(value) for value in input_dims)
    )


def postprocess(output, original_shape, min_depth, max_depth):
    depth_map = np.asarray(output, dtype=np.float32)
    depth_map = np.squeeze(depth_map)

    if depth_map.ndim != 2:
        raise RuntimeError(
            f"Expected a single 2D depth map after squeeze, "
            f"got shape {depth_map.shape}"
        )

    original_height, original_width = original_shape[:2]
    depth_map = cv2.resize(
        depth_map,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )

    depth_map = np.nan_to_num(
        depth_map,
        nan=min_depth,
        posinf=max_depth,
        neginf=min_depth
    )
    depth_map = np.clip(depth_map, min_depth, max_depth)

    return depth_map.astype(np.float32)


def colorize_depth(depth_map):
    valid_mask = np.isfinite(depth_map)

    if not np.any(valid_mask):
        normalized = np.zeros(depth_map.shape, dtype=np.uint8)
    else:
        valid_values = depth_map[valid_mask]
        depth_min = float(valid_values.min())
        depth_max = float(valid_values.max())

        if depth_max <= depth_min:
            normalized = np.zeros(depth_map.shape, dtype=np.uint8)
        else:
            normalized = (
                (depth_max - depth_map) /
                (depth_max - depth_min) *
                255.0
            )
            normalized = np.clip(
                normalized,
                0,
                255
            ).astype(np.uint8)

    return cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_INFERNO
    )


def main():
    parser = argparse.ArgumentParser(description="DINOv2 ViT-B14 NYU DPT ADLA Demo")
    parser.add_argument("--backbone", required=True, help="Path to the four-output DINOv2 depth backbone .adla")
    parser.add_argument("--depth", required=True, help="Path to the single-input DINOv2 depth head .adla")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--output-dir", default="depth_results", help="Directory for depth outputs")
    parser.add_argument("--min-depth", type=float, default=0.001, help="Minimum depth")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Maximum NYU depth")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    backbone = AMLNN()
    backbone.init_runtime(mode="native", enable_perf=True)
    backbone.load_model(path=args.backbone)
    backbone_info = backbone.get_tensor_info()

    depth_model = AMLNN()
    depth_model.init_runtime(mode="native", enable_perf=True)
    depth_model.load_model(path=args.depth)
    depth_info = depth_model.get_tensor_info()

    print(backbone.get_sdk_version())

    image_files = []
    for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, extension.upper())))

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        backbone.uninit()
        depth_model.uninit()
        return 0

    if len(backbone_info["outputs"]) != 4:
        raise RuntimeError(f"Expected 4 depth backbone outputs, got {len(backbone_info['outputs'])}")
    if len(depth_info["inputs"]) != 1:
        raise RuntimeError(f"Expected 1 depth-head input, got {len(depth_info['inputs'])}")
    if len(depth_info["outputs"]) != 1:
        raise RuntimeError(f"Expected 1 depth-head output, got {len(depth_info['outputs'])}")

    print(f"Found {len(image_files)} image file(s) to process:")
    for image_file in image_files:
        print(f"  - {os.path.basename(image_file)}")
    print()

    backbone_input_attr = backbone_info["inputs"][0]
    backbone_input_shape = (int(backbone_input_attr["dims"][1]), int(backbone_input_attr["dims"][2]))
    backbone_input_scale = float(backbone_input_attr["scale"])
    backbone_input_zero_point = int(backbone_input_attr["zp"])
    backbone_input_type = int(backbone_input_attr["type"])

    depth_input_attr = depth_info["inputs"][0]
    depth_input_dims = tuple(int(value) for value in depth_input_attr["dims"])
    print(depth_input_dims)
    depth_input_scale = float(depth_input_attr["scale"])
    depth_input_zero_point = int(depth_input_attr["zp"])
    depth_input_type = int(depth_input_attr["type"])

    print(f"Backbone input: name={backbone_input_attr['name']}, shape={backbone_input_attr['dims']}")
    for index, output in enumerate(backbone_info["outputs"]):
        print(f"Backbone output {index}: name={output['name']}, shape={output['dims']}")
    print(f"Depth input: name={depth_input_attr['name']}, shape={depth_input_attr['dims']}")
    print(f"Depth output: name={depth_info['outputs'][0]['name']}, shape={depth_info['outputs'][0]['dims']}")
    print()

    for index, image_path in enumerate(image_files, 1):
        print("=" * 60)
        print(f"Processing image {index}/{len(image_files)}: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            backbone_input, original_img = preprocess(
                image_path,
                backbone_input_shape,
                backbone_input_scale,
                backbone_input_zero_point,
                backbone_input_type
            )

            backbone_outputs = backbone.inference(
                inputs=[backbone_input]
            )

            backbone_features = []

            for output_index, output in enumerate(backbone_outputs):
                feature = np.asarray(output, dtype=np.float32)
                backbone_features.append(feature)

                print(
                    f"      Backbone output {output_index}: "
                    f"shape={feature.shape}, "
                    f"min={float(feature.min()):.6f}, "
                    f"max={float(feature.max()):.6f}, "
                    f"mean={float(feature.mean()):.6f}, "
                    f"std={float(feature.std()):.6f}"
                )

            concat_features = np.concatenate(backbone_features, axis=2)
            depth_input_float = np.transpose(concat_features, (0, 2, 3, 1))

            concat_features = np.concatenate([
                np.asarray(backbone_outputs[0], dtype=np.float32),
                np.asarray(backbone_outputs[1], dtype=np.float32),
                np.asarray(backbone_outputs[2], dtype=np.float32),
                np.asarray(backbone_outputs[3], dtype=np.float32)
            ], axis=2).astype(np.float32)

            depth_input_float = reshape_for_model_input(
                concat_features,
                depth_input_dims
            )

            depth_input = prepare_input_tensor(
                depth_input_float,
                depth_input_scale,
                depth_input_zero_point,
                depth_input_type
            )

            depth_outputs = depth_model.inference(
                inputs=[depth_input]
            )

            depth_map = postprocess(
                depth_outputs[0],
                original_img.shape,
                args.min_depth,
                args.max_depth
            )

            image_stem = os.path.splitext(
                os.path.basename(image_path)
            )[0]
            depth_png_path = os.path.join(
                args.output_dir,
                f"{image_stem}_depth.png"
            )

            depth_color = colorize_depth(depth_map)
            cv2.imwrite(depth_png_path, depth_color)

            print("    Results:")
            print(f"      Depth shape: {depth_map.shape}")
            print(f"      Depth range: {float(depth_map.min()):.6f} to {float(depth_map.max()):.6f}")
            print(f"      Mean depth: {float(depth_map.mean()):.6f}")
            print(f"      Visualization: {depth_png_path}")

        except Exception as error:
            print(f"Error processing {os.path.basename(image_path)}: {error}")

        print()

    print("=" * 60)
    print("Backbone performance:")
    print(backbone.get_perf_info())
    print("Depth performance:")
    print(depth_model.get_perf_info())

    backbone.uninit()
    depth_model.uninit()


if __name__ == "__main__":
    main()
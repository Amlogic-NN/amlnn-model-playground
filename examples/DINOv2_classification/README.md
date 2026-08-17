# Dinov2 Linear Classifier

This example runs Dinov2 Linear Classifier with AMLNN. The full flow is:

1. Prepare or download the ONNX models.
2. Convert the ONNX models to ADLA models.
3. Run the Python demo with the ADLA models.
4. Run the C++ (Linux/Android) demo with the ADLA models.
5. Check classification results.

## Directory Layout

```bash
examples/DINOv2_classification/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and ImageNet labels
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## License

This example code is licensed under the Apache License 2.0. See the repository root [LICENSE](../../LICENSE) file for the complete license terms.

The dinov2 ViT-S/14 backbone and ImageNet-1k linear classification head used in this example originate from the [Meta AI dinov2 project](https://github.com/facebookresearch/dinov2). The original dinov2 code and model weights are released under the Apache License 2.0.

The ONNX and ADLA model files distributed for this example are converted and, where applicable, compiled or quantized forms of the original dinov2 model weights. These model files have been modified from the original distribution through model export, graph conversion, model separation, compilation, and/or quantization.

The converted model files are redistributed under the Apache License 2.0. When redistributing these files, you must:

* provide recipients with a copy of the Apache License 2.0;
* retain applicable copyright, patent, trademark, and attribution notices;
* preserve this notice or an equivalent notice identifying the original dinov2 project; and
* clearly state that the model files were converted or otherwise modified from the original model weights.

> Copyright (c) Meta Platforms, Inc. and affiliates.
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
> ```
> http://www.apache.org/licenses/LICENSE-2.0
> ```
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.

The preprocessing, inference, and postprocessing implementation in this example was developed for this project with AI assistance and was subsequently reviewed and adapted by the project maintainers.

## 1. Prepare The ONNX Models

### Download ONNX

Download the prepared DINOv2 backbone and linear classifier ONNX models:

### [Download DINOv2 Backbone ONNX model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dinov2_lc/dinov2_vits14_backbone_sim.onnx)

### [Download DINOv2 Linear Classification ONNX model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dinov2_lc/dinov2_vits14_linear4_head_sim.onnx)

Download the ImageNet class names:

### [Download ImageNet labels here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/mobilenet/labels.txt)

Place the downloaded models under `examples/DINOv2_classification/model/` and the labels file under `examples/DINOv2_classification/input/`.

Expected paths:

```text
examples/DINOv2_classification/model/dinov2_vits14_backbone_sim.onnx
examples/DINOv2_classification/model/dinov2_vits14_linear4_head_sim.onnx
examples/DINOv2_classification/input/labels.txt
```

## 2. Convert ONNX To ADLA

Run the ADLA export script from `examples/DINOv2_classification/py`:

```bash
cd examples/DINOv2_classification/py
python export_adla.py \
  --backbone-onnx ../model/dinov2_vits14_backbone_sim.onnx \
  --classifier-onnx ../model/dinov2_vits14_linear4_head_sim.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| Parameter           | Description                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--backbone-onnx`   | Path to the DINOv2 backbone `.onnx` model.                                                      |
| `--classifier-onnx` | Path to the DINOv2 linear classifier `.onnx` model.                                             |
| `--target-platform` | Target platform ID. See the full list of supported platforms [**HERE**](../../docs/mapping.md). |
| `--output-dir`      | (Optional) Directory where the generated `.adla` models will be saved. Defaults to `../model`.  |

> **Quantization note:** The current backbone and linear classifier configurations use `w8a16` with `activation_dtype="f16"`. A calibration dataset is therefore not required.

After conversion, AMLNN's generated filenames are preserved.

Expected model paths:

```text
examples/DINOv2_classification/model/dinov2_vits14_backbone_sim_w8a16.adla
examples/DINOv2_classification/model/dinov2_vits14_linear4_head_sim_w8a16.adla
```

## 3. Run Python Demo

### Prerequisites

* Python 3.10
* Required packages: `amlnn`

### Install Dependencies

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### Run on Device

```bash
python dinov2_lc_inference.py \
    --backbone ../model/dinov2_vits14_backbone_sim_w8a16.adla \
    --classifier ../model/dinov2_vits14_linear4_head_sim_w8a16.adla \
    --image-dir ../input \
    --labels ../input/labels.txt \
    --topk 5
```

Argument Descriptions:

| Argument             | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `--backbone`   | Path to the compiled DINOv2 backbone `.adla` model.                    |
| `--classifier` | Path to the compiled DINOv2 linear classifier `.adla` model.           |
| `--image-dir`        | Directory containing test images.                                      |
| `--labels`           | Path to the ImageNet class names `.txt` file.                          |
| `--topk`             | (Optional) Number of classification results to print. Defaults to `5`. |

The script will automatically process all image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) in the specified image directory.

## 4. Run C++ Demo

### Build For Android

#### Prerequisites

* **Android NDK** (r27d recommended) installed on your system.
* **AMLNN Toolkit** downloaded and extracted.
* Prebuilt OpenCV for Android located in the `dependency/opencv/` folder.

#### 1. Setup Environment

Export the paths to your NDK (the toolchain) and AMLNN (the neural network dependency) so the script can find them.

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. Build

Navigate to the C++ directory and run the build script.

```bash
cd examples/DINOv2_classification/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

The executable will be generated in the build folder corresponding to the selected Android ABI:

* 64-bit: `build/android/arm64-v8a/dinov2_lc_demo`
* 32-bit: `build/android/armeabi-v7a/dinov2_lc_demo`

#### 3. Example Run

The following example uses the default 64-bit (`arm64-v8a`) build.

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/dinov2_lc_demo /data/local/tmp/
adb push ../model/dinov2_vits14_backbone_sim_w8a16.adla /data/local/tmp/
adb push ../model/dinov2_vits14_linear4_head_sim_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dinov2_lc_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./dinov2_lc_demo <backbone.adla> <classifier.adla> <image_dir> <labels.txt> [topk]
./dinov2_lc_demo dinov2_vits14_backbone_sim_w8a16.adla dinov2_vits14_linear4_head_sim_w8a16.adla input/ input/labels.txt 5
```

> **Note:** For a 32-bit (`armeabi-v7a`) build, use `build/android/armeabi-v7a/dinov2_lc_demo` and the corresponding 32-bit library path. Replace the `.adla` filenames with your actual generated model filenames.

---

### Build For Linux

The Linux build process supports two distinct modes:

1. **Standard Linux cross-compilation** (default)
2. **Yocto SDK compilation**

#### Mode 1: Standard Linux Cross-Compile (Default)

##### Prerequisites

* A GCC Cross-Compiler toolchain (GCC 10.3 recommended).
* The toolchain's `bin/` folder must be added to your system's `PATH`.
* Prebuilt OpenCV located in the `dependency/opencv/` folder.
* `AMLNN_HOME` environment variable set.

##### 1. Setup Environment

Add your downloaded toolchain to your `PATH` and export the `AMLNN_HOME` variable so the script can find the compiler and neural network dependencies.

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds, add the 64-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# OR for 32-bit (arm) builds, add the 32-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

##### 2. Build

```bash
cd examples/DINOv2_classification/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **Optional Override:** If your compiler has a different prefix name (for example, `aarch64-linux-gnu` instead of `aarch64-none-linux-gnu`), you can override the default by setting the `GCC_COMPILER` variable:

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/linux/64/dinov2_lc_demo`
* 32-bit: `build/linux/32/dinov2_lc_demo`

#### Mode 2: Yocto/Debian/Armbian Build

##### 1. Prerequisites

* Yocto SDK installed.
* CMake Toolchain file available.
* Prebuilt OpenCV located at `../../../dependency/opencv/` (relative to the script directory).

##### 2. Setup Environment

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. Build

```bash
cd examples/DINOv2_classification/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **Note:** You can also use the `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` environment variables instead of passing the `-s` and `-t` flags.

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/yocto/64/dinov2_lc_demo`
* 32-bit: `build/yocto/32/dinov2_lc_demo`

#### 3. Example Run

The following example uses the default 64-bit Linux build.

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/dinov2_lc_demo /data/local/tmp/
adb push ../model/dinov2_vits14_backbone_sim_w8a16.adla /data/local/tmp/
adb push ../model/dinov2_vits14_linear4_head_sim_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dinov2_lc_demo

# Usage: ./dinov2_lc_demo <backbone.adla> <classifier.adla> <image_dir> <labels.txt> [topk]
./dinov2_lc_demo dinov2_vits14_backbone_sim_w8a16.adla dinov2_vits14_linear4_head_sim_w8a16.adla input/ input/labels.txt 5
```

> **Note:** Replace the `.adla` filenames with your actual generated model filenames. Adjust the executable path if using a 32-bit or Yocto build.

## 5. Results

### Performance Feedback

By setting the log level to `INFO`, the program provides runtime performance information after inference. The console output may include:

* **Hardware Information:** System and ADLA library versions.
* **Model Overview:** Input and output tensor configurations.
* **NPU Metrics:** Inference latency and DRAM bandwidth usage.

### Classification Output

For each input image, the demo runs the DINOv2 backbone followed by the linear classifier and prints the Top-K classification results with their respective probabilities.

Example:

```text
============================================================
Processing image 1/5: dog_224x224.jpg
============================================================
I amlnn_inference_base amlnn_inference_breakdown_ms: input_prepare=0.028, runtime_inference=55.280, output_postprocess=0.031, total=55.343
I amlnn_inference_base amlnn_inference_breakdown_ms: input_prepare=0.033, runtime_inference=2.660, output_postprocess=0.023, total=2.720
    Results:
      1. Pekinese (0.9685)
      2. Maltese dog (0.0246)
      3. West Highland white terrier (0.0063)
      4. Chihuahua (0.0002)
      5. tiger cat (0.0001)
============================================================
```

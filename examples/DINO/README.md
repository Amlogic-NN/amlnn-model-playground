# DINO

This example runs DINO with AMLNN. The full flow is:

1. Prepare or download the ONNX models.
2. Convert the ONNX models to ADLA models.
3. Run the Python demo with the ADLA models.
4. Run the C++ (Linux/Android) demo with the ADLA models.
5. Check classification results.

## Directory Layout

```bash
examples/DINO/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and ImageNet labels
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## License

This example code is licensed under the Apache License 2.0. See the repository root [LICENSE](../../LICENSE) file for details.

The DINO ViT-S/16 backbone and ImageNet linear classifier weights used in this example are provided by a third party and originate from the [Facebook Research DINO project](https://github.com/facebookresearch/dino). The original DINO project and released model weights are distributed under the Apache License 2.0.

The converted and quantized DINO backbone and linear classifier models used by this example are distributed from our server under the Apache License 2.0. Please retain the applicable copyright, license, and attribution notices when redistributing the models.

> Copyright (c) Facebook, Inc. and its affiliates.
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

The model preprocessing and postprocessing code in this example was partially generated with AI assistance and subsequently reviewed and adapted for this project. If any part is inadvertently similar to existing work and causes concern, please contact us, and we will remove or adjust it promptly.

## 1. Prepare The ONNX Models

### Download ONNX

Download the prepared DINO backbone and linear classifier ONNX models:

### [Download DINO Backbone ONNX model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dino/dino_backbone.onnx)

### [Download DINO Linear Classification ONNX model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dino/dino_linear.onnx)

Download the ImageNet class names:

### [Download ImageNet labels here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/mobilenet/labels.txt)

Place the downloaded models under `examples/DINO/model/` and the labels file under `examples/DINO/input/`.

Expected paths:

```text
examples/DINO/model/dino_backbone.onnx
examples/DINO/model/dino_linear.onnx
examples/DINO/input/labels.txt
```

## 2. Convert ONNX To ADLA

Run the ADLA export script from `examples/DINO/py`:

```bash
cd examples/DINO/py
python export_adla.py \
  --backbone-onnx ../model/dino_backbone.onnx \
  --classifier-onnx ../model/dino_linear.onnx \
  --dataset-path ../../../resource/classification_subset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| Parameter           | Description                                                                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--backbone-onnx`   | Path to the DINO backbone `.onnx` model.                                                                                                                           |
| `--classifier-onnx` | Path to the DINO linear classifier `.onnx` model.                                                                                                                  |
| `--dataset-path`    | (Optional) Path to a `.txt` file containing the input paths used for BACKBONE quantization calibration. Required for the current `w8a8` backbone configuration.[1] |
| `--target-platform` | Target platform ID. See the full list of supported platforms [**HERE**](../../docs/mapping.md).                                                                    |
| `--output-dir`      | (Optional) Directory where the generated `.adla` models will be saved. Defaults to `../model`.                                                                     |

> **[1] Quantization dataset note**
>
> A calibration dataset is **not required** when using `activation_dtype="f16"`, such as `w8a16 (f16)` or `w16a16 (f16)`. Set this explicitly in the `amlnn.config()` call.
>
> Without `activation_dtype="f16"`, `w8a16` and `w16a16` use `i16` activations and require a calibration dataset. If no dataset is provided, AMLNN generates random calibration data instead of failing, which can severely degrade model accuracy.
>
> `activation_dtype` does not apply to `w8a8`.
>
> In the current DINO export configuration, the backbone uses `w8a8` and requires the calibration dataset, while the linear classifier uses `w8a16 (f16)` and does not require one.

After conversion, AMLNN's generated filenames are preserved.

Expected model paths:

```text
examples/DINO/model/dino_backbone_w8a8.adla
examples/DINO/model/dino_linear_w8a16.adla
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
python dino_inference.py \
    --backbone ../model/dino_backbone_w8a8.adla \
    --classifier ../model/dino_linear_w8a16.adla \
    --image-dir ../input \
    --labels ../input/labels.txt \
    --topk 5
```

Argument Descriptions:

| Argument             | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `--backbone`   | Path to the compiled DINO backbone `.adla` model.                      |
| `--classifier` | Path to the compiled DINO linear classifier `.adla` model.             |
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
cd examples/DINO/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

The executable will be generated in the build folder corresponding to the selected Android ABI:

* 64-bit: `build/android/arm64-v8a/dino_demo`
* 32-bit: `build/android/armeabi-v7a/dino_demo`

#### 3. Example Run

The following example uses the default 64-bit (`arm64-v8a`) build.

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/dino_demo /data/local/tmp/
adb push ../model/dino_backbone_w8a8.adla /data/local/tmp/
adb push ../model/dino_linear_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dino_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./dino_demo <backbone.adla> <classifier.adla> <image_dir> <labels.txt> [topk]
./dino_demo dino_backbone_w8a8.adla dino_linear_w8a16.adla input/ input/labels.txt 5
```

> **Note:** For a 32-bit (`armeabi-v7a`) build, use `build/android/armeabi-v7a/dino_demo` and the corresponding 32-bit library path. Replace the `.adla` filenames with your actual generated model filenames.

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
cd examples/DINO/cpp

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

* 64-bit: `build/linux/64/dino_demo`
* 32-bit: `build/linux/32/dino_demo`

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
cd examples/DINO/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **Note:** You can also use the `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` environment variables instead of passing the `-s` and `-t` flags.

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/yocto/64/dino_demo`
* 32-bit: `build/yocto/32/dino_demo`

#### 3. Example Run

The following example uses the default 64-bit Linux build.

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/dino_demo /data/local/tmp/
adb push ../model/dino_backbone_w8a8.adla /data/local/tmp/
adb push ../model/dino_linear_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dino_demo

# Usage: ./dino_demo <backbone.adla> <classifier.adla> <image_dir> <labels.txt> [topk]
./dino_demo dino_backbone_w8a8.adla dino_linear_w8a16.adla input/ input/labels.txt 5
```

> **Note:** Replace the `.adla` filenames with your actual generated model filenames. Adjust the executable path if using a 32-bit or Yocto build.

## 5. Results

### Performance Feedback

By setting the log level to `INFO`, the program provides runtime performance information after inference. The console output may include:

* **Hardware Information:** System and ADLA library versions.
* **Model Overview:** Input and output tensor configurations.
* **NPU Metrics:** Inference latency and DRAM bandwidth usage.

### Classification Output

For each input image, the demo runs the DINO backbone followed by the linear classifier and prints the Top-K classification results with their respective probabilities.

Example:

```text
============================================================
Processing image: "fish_224x224.jpeg"
============================================================
Backbone inference time: 18.1208 ms
Classifier inference time: 0.589565 ms

    Top-5 Results:
      1. tench                 prob=0.825604
      2. neck brace            prob=0.048056
      3. shower cap            prob=0.046215
      4. eel                   prob=0.006896
      5. planetarium           prob=0.005853

============================================================
```
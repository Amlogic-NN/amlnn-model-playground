# EVA02

This example runs EVA02 with AMLNN. The full flow is:

1. Prepare or download an ONNX model.
2. Convert the ONNX model to an ADLA model.
3. Run the Python demo with the ADLA model.
4. Run the C++ (Linux/Android) demo with the ADLA model.
5. Check classification results.

## Directory Layout

```bash
examples/eva02/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and ImageNet labels
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## License

This example code is licensed under the Apache License 2.0. See the repository root [LICENSE](../../LICENSE) file for details.

The EVA-02 model and related source code used by this example are provided by a third party and originate from the [EVA-02 repository](https://github.com/baaivision/EVA). They are distributed under the MIT License. When redistributing the EVA-02 model, source code, or substantial portions derived from them, retain the following copyright and license notice:

> MIT License
>
> Copyright (c) 2022 BAAI-Vision
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Review and comply with any additional usage terms associated with the specific pretrained checkpoint or dataset before redistribution or commercial use.

Parts of the model preprocessing and postprocessing code in this example were developed with AI assistance. Please report any suspected licensing or attribution issue so that it can be reviewed and corrected.

## 1. Prepare The ONNX Model

### Download ONNX

Download the prepared EVA02 ONNX model:

### [Download EVA02 ONNX model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/EVA02/eva02_base_sim.onnx)

Download the ImageNet class names:

### [Download ImageNet labels here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/mobilenet/labels.txt)

Place the downloaded ONNX model under `examples/eva02/model/` and the labels file under `examples/eva02/input/`.

Expected paths:

```text
examples/eva02/model/eva02_base_sim.onnx
examples/eva02/input/labels.txt
```

## 2. Convert ONNX To ADLA

Run the ADLA export script from `examples/eva02/py`:

```bash
cd examples/eva02/py
python export_adla.py \
  --onnx ../model/eva02_base_sim.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| Parameter           | Description                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | Path to the EVA02 `.onnx` model.                                                                |
| `--target-platform` | Target platform ID. See the full list of supported platforms [**HERE**](../../docs/mapping.md). |
| `--output-dir`      | (Optional) Directory where the generated `.adla` model will be saved. Defaults to `../model`.   |


After conversion, AMLNN's generated filename is preserved.

Expected model path:

```text
examples/eva02/model/eva02_base_sim_w16a16.adla
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
python eva02_inference.py \
    --adla ../model/eva02_base_sim_w16a16.adla \
    --image-dir ../input \
    --labels ../input/labels.txt
```

Argument Descriptions:

| Argument      | Description                                   |
| ------------- | --------------------------------------------- |
| `--adla`      | Path to the EVA02 `.adla` model.              |
| `--image-dir` | Directory containing test images.             |
| `--labels`    | Path to the ImageNet class names `.txt` file. |

The script will automatically process all image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) in the specified image directory and save the classification result images to the model result directory.

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
cd examples/eva02/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

The executable will be generated in the build folder corresponding to the selected Android ABI:

* 64-bit: `build/android/arm64-v8a/eva02_demo`
* 32-bit: `build/android/armeabi-v7a/eva02_demo`

#### 3. Example Run

The following example uses the default 64-bit (`arm64-v8a`) build.

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/eva02_demo /data/local/tmp/
adb push ../model/eva02_base_sim_w16a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x eva02_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./eva02_demo <model_path> <image_dir> <labels.txt>
./eva02_demo eva02_base_sim_w16a16.adla input/ input/labels.txt
```

> **Note:** For a 32-bit (`armeabi-v7a`) build, use `build/android/armeabi-v7a/eva02_demo` and the corresponding 32-bit library path. Replace the `.adla` filename with your actual generated model filename.

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
cd examples/eva02/cpp

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

* 64-bit: `build/linux/64/eva02_demo`
* 32-bit: `build/linux/32/eva02_demo`

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
cd examples/eva02/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **Note:** You can also use the `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` environment variables instead of passing the `-s` and `-t` flags.

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/yocto/64/eva02_demo`
* 32-bit: `build/yocto/32/eva02_demo`

#### 3. Example Run

The following example uses the default 64-bit Linux build.

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/eva02_demo /data/local/tmp/
adb push ../model/eva02_base_sim_w16a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x eva02_demo

# Usage: ./eva02_demo <model_path> <image_dir> <labels.txt>
./eva02_demo eva02_base_sim_w16a16.adla input/ input/labels.txt
```

> **Note:** Replace the `.adla` filename with your actual generated model filename. Adjust the executable path if using a 32-bit or Yocto build.

## 5. Results

### Performance Feedback

By setting the log level to `INFO`, the program provides runtime performance information after inference. The console output may include:

* **Hardware Information:** System and ADLA library versions.
* **Model Overview:** Input and output tensor configurations.
* **NPU Metrics:** Inference latency and DRAM bandwidth usage.

### Classification Output

For each input image, the program prints the Top-5 classification results with their respective scores.

Example:

```text
============================================================
Processing image: "fox.jpg"
============================================================
Inference time: 526.8334 ms
Top 5 results:
  1. kit fox (0.7245)
  2. red fox (0.1264)
  3. grey fox (0.0080)
  4. Arctic fox (0.0024)
  5. coyote (0.0021)
============================================================
```

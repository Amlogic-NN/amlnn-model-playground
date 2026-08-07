# Lite_transformer

This example runs Lite_transformer with AMLNN. The full flow is:

1. Prepare or download an ONNX model.
2. Convert the ONNX model to an ADLA model.
3. Run the Python demo with ADLA model.
4. Run the C++ (Linux/Android) demo with the ADLA model.
5. Check detection images/results.

## Directory layout
```bash
examples/lite_transformer/
├── assets/            # Necessary files for pre and postprocessing
├── cpp/               # C++ demo and build scripts
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## License

This example code is licensed under the Apache License 2.0. See the repository root [LICENSE](../../LICENSE) file for details.

The Lite Transformer model architecture, pretrained model assets, vocabulary dictionaries, and BPE codes used in this example originate from the Lite Transformer and Fairseq projects. The Lite Transformer software is distributed under the BSD 3-Clause License, while Fairseq and its applicable model assets are distributed under the MIT License.

The converted ONNX and quantized ADLA Lite Transformer models used by this example are derived from these third-party assets. When redistributing the models or accompanying text-processing assets, please retain the applicable copyright, license, disclaimer, and attribution notices.

The complete third-party license texts and attribution notices are available in [THIRD_PARTY_NOTICES.txt](./THIRD_PARTY_NOTICES.txt).

The model preprocessing and postprocessing code in this example was partially generated with AI assistance and subsequently reviewed and adapted for this project. If any part is inadvertently similar to existing work and causes concern, please contact us, and we will remove or adjust it promptly.

## 1. Prepare The ONNX Model

### Download onnx

Download the prepared onnx model and put it under `examples/lite_transformer/model/`:

[Download lite_transformer ONNX model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/Lite_transformer/lite_transformer_en_fr.onnx)

[Download lite_transformer assets.zip here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/Lite_transformer/assets.zip)

Place the downloaded onnx files under `examples/lite_transformer/model/`

## 2. Model Conversion
The model conversion is done using the export_adla.py script.
```bash
cd py
Usage:   python export_adla.py --onnx ../model/lite_transformer_en_fr.onnx \
                               --target-platform 007
```

| Parameter           | Description                                                                                                                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--onnx`   | lite_transformer backbone `.onnx` model path                                                                                                                                                                                                        |
| `--target-platform` | Specify target platform. For specific platforms, click [**HERE**](../../docs/mapping.md) to see the full list                                                                                                                           |
| `--adla`            | Output directory for the generated backbone and classifier `.adla` files. Optional; defaults to `../model` if not specified.                                                                                                            |


## 3. Run Python Demo

**Prerequisites:**
- Python 3.10
- Required packages: `amlnn`, `subword-nmt`, `sacremoses`

**Install dependencies:**
```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl subword-nmt sacremoses
```

**Run on device:**
```bash
python lite_inference.py \
    --model-path ../model/lite_transformer_en_fr.adla \
    --assets ../assets \
    --texts "Hello world." "This is a translation test."
```

Argument Descriptions:

| Argument       | Description                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------- |
| `--model-path` | Path to the Lite Transformer English-to-French `.adla` model.                               |
| `--assets-dir` | Directory containing `dict.en.txt`, `dict.fr.txt`, and `bpecodes`.                          |
| `--texts`      | One or more English sentences to translate. Wrap each complete sentence in quotation marks. |


The script will automatically process all image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) in the current directory and save results to a `{model_name}_result` folder.

## 4. Run C++ Demo

### Build For Android

**Prerequisites:**
- **Android NDK** (r27d recommended) installed on your system.
- **AMLNN Toolkit** downloaded and extracted.
- Prebuilt OpenCV for Android located in the `dependency/opencv/` folder.

**1. Setup Environment:**
Export the paths to your NDK (the toolchain) and AMLNN (the neural network dependency) so the script can find them.
```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

**2. Build:**
Navigate to the C++ directory and run the build script.

```bash
cd examples/lite_transformer/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

The executable will be generated at `build/android/lite_demo` (Note: executable name may vary, verify in build folder).

**Run:**

```bash
# Push executable to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/lite_demo /data/local/tmp/
adb push ../model/lite_transformer_en_fr.adla /data/local/tmp/
adb push ../assets/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x lite_demo
export LD_LIBRARY_PATH=/vendor/lib64 or (/vendor/lib)

# Usage: ./lite_demo <model.adla> <en-fr_text_assets> [--max-new-tokens N](Optional) <texts ...>
./lite_demo lite_transformer_en_fr.adla assets/ --max-new-tokens 64 "This is a translation test."
```

**Note:** Replace `lite_transformer.adla` with your actual model file path.

---
### Build For Linux

The Linux build process supports two distinct modes: **Standard Linux cross-compilation** (default) and **Yocto SDK compilation**.

### Mode 1: Standard Linux Cross-Compile (Default)

**Prerequisites:**
- A GCC Cross-Compiler toolchain (GCC 10.3 recommended).
- The toolchain's `bin/` folder must be added to your system's `PATH`.
- Prebuilt OpenCV located in the `dependency/opencv/` folder.
- `AMLNN_HOME` environment variable set

**1. Setup Environment:**
Add your downloaded toolchain to your `PATH` and export the `AMLNN_HOME` variable so the script can find the compiler and neural network dependencies.
```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds, add the 64-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# OR for 32-bit (arm) builds, add the 32-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

**2. Build:**
```bash
cd examples/lite_transformer/cpp
# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

*(Optional Override):* If your compiler has a different prefix name (for example, `aarch64-linux-gnu` instead of `aarch64-none-linux-gnu`), you can override the default by setting the `GCC_COMPILER` variable:
```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

The executable will be generated at `build/linux/64/lite_demo` (or `build/linux/32/lite_demo`).

### Mode 2: Yocto/Debian/Armbian Build

**Prerequisites:**
- Yocto SDK installed
- CMake Toolchain file available
- Prebuilt OpenCV located at `../../../dependency/opencv/` (relative to the script directory)

**Build:**
```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/lite_transformer/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Or build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```
*(Note: You can also use the `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` environment variables instead of passing the `-s` and `-t` flags).*

The executable will be generated at `build/yocto/64/lite_demo` (or `build/yocto/32/lite_demo`).

---

**Run:**

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/lite_demo /data/local/tmp/
adb push ../model/lite_transformer_en_fr.adla /data/local/tmp/
adb push ../assets/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x lite_demo

# Usage: ./lite_demo <model.adla> <en-fr_text_assets> [--max-new-tokens N](Optional) <texts ...>
./lite_demo lite_transformer_en_fr.adla assets/ --max-new-tokens 64 "This is a translation test."
```

**Note:** Replace with your actual model file name.

## 5. Results

**Performance Feedback**

By setting the loglevel to INFO, the program provides real-time performance metrics upon completion. The console log will display essential hardware and execution details, including:
- Hardware Information: System and ADLA library versions.
- Model Overview: Basic input/output configurations.
- NPU Metrics: Total inference time (latency) and total DRAM bandwidth consumption.

**Translation Output**

```bash
============================================================
Translating text 1/1: Hello this is a translation test. I am holding a banana while jumping.
============================================================
Tokenized input: Hello this is a translation test . I am holding a banana while jumping .
BPE input: H@@ ello this is a translation test . I am holding a ban@@ ana while jum@@ ping .
Source token IDs: [329, 14048, 70, 37, 18, 7183, 1609, 7, 99, 1156, 5289, 18, 5614, 3570, 826, 14603, 10815, 7, 2]
Generated token IDs: [9803, 392, 4, 812, 41, 33, 1609, 5, 6012, 4, 470, 2582, 25, 3611, 5, 20942, 35, 5614, 3528, 221, 25, 411, 10346, 301, 7]
Generated BPE: Bon@@ jour , c&apos; est un test de traduction , je suis en train de détenir une ban@@ ane tout en s@@ aut@@ ant .
Translation: Bonjour, c'est un test de traduction, je suis en train de détenir une banane tout en sautant.

============================================================
```


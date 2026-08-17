# Lite Transformer

This example runs Lite Transformer English-to-French translation with AMLNN. The full flow is:

1. Prepare or download the ONNX model and text-processing assets.
2. Convert the ONNX model to an ADLA model.
3. Run the Python demo with the ADLA model.
4. Run the C++ (Linux/Android) demo with the ADLA model.
5. Check translation results.

## Directory Layout

```bash
examples/lite_transformer_en_fr/
├── assets/            # Vocabulary dictionaries and BPE codes
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

### Download ONNX

Download the prepared Lite Transformer English-to-French ONNX model:

### [Download Lite Transformer ONNX model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/Lite_transformer/lite_transformer_en_fr.onnx)

Download the required vocabulary and BPE assets:

### [Download Lite Transformer assets.zip here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/Lite_transformer/assets.zip)

Place the downloaded ONNX model under `examples/lite_transformer_en_fr/model/` and extract the assets under `examples/lite_transformer_en_fr/assets/`.

Expected paths:

```text
examples/lite_transformer_en_fr/model/lite_transformer_en_fr.onnx
examples/lite_transformer_en_fr/assets/
├── dict.en.txt
├── dict.fr.txt
└── bpecodes
```

## 2. Convert ONNX To ADLA

Run the ADLA export script from `examples/lite_transformer_en_fr/py`:

```bash
cd examples/lite_transformer_en_fr/py
python export_adla.py \
  --onnx ../model/lite_transformer_en_fr.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| Parameter           | Description                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | Path to the Lite Transformer `.onnx` model.                                                     |
| `--target-platform` | Target platform ID. See the full list of supported platforms [**HERE**](../../docs/mapping.md). |
| `--output-dir`      | (Optional) Directory where the generated `.adla` model will be saved. Defaults to `../model`.   |

After conversion, AMLNN's generated filename is preserved.

Expected model path:

```text
examples/lite_transformer_en_fr/model/lite_transformer_en_fr_w8a16.adla
```

## 3. Run Python Demo

### Prerequisites

* Python 3.10
* Required packages: `amlnn`, `subword-nmt`, `sacremoses`

### Install Dependencies

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl subword-nmt sacremoses
```

### Run on Device

```bash
python lite_inference.py \
    --adla ../model/lite_transformer_en_fr_w8a16.adla \
    --assets ../assets \
    --texts "Hello world." "This is a translation test."
```

Argument Descriptions:

| Argument   | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| `--adla`   | Path to the Lite Transformer English-to-French `.adla` model.                               |
| `--assets` | Directory containing `dict.en.txt`, `dict.fr.txt`, and `bpecodes`.                          |
| `--texts`  | One or more English sentences to translate. Wrap each complete sentence in quotation marks. |

The script processes each supplied English sentence and prints the corresponding French translation to the console.

## 4. Run C++ Demo

### Build For Android

#### Prerequisites

* **Android NDK** (r27d recommended) installed on your system.
* **AMLNN Toolkit** downloaded and extracted.
* Prebuilt OpenCV for Android located in the `dependency/opencv/` folder.

#### 1. Setup Environment

Export the paths to your NDK and AMLNN so the build script can find the required toolchain and neural network dependencies.

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. Build

```bash
cd examples/lite_transformer_en_fr/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

The executable will be generated in the build folder corresponding to the selected Android ABI:

* 64-bit: `build/android/arm64-v8a/lite_demo`
* 32-bit: `build/android/armeabi-v7a/lite_demo`

#### 3. Example Run

The following example uses the default 64-bit (`arm64-v8a`) build.

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/lite_demo /data/local/tmp/
adb push ../model/lite_transformer_en_fr_w8a16.adla /data/local/tmp/
adb push ../assets/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x lite_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./lite_demo <model.adla> <en-fr_text_assets> [--max-new-tokens N] <texts ...>
./lite_demo lite_transformer_en_fr_w8a16.adla assets/ --max-new-tokens 64 "This is a translation test."
```

> **Note:** For a 32-bit (`armeabi-v7a`) build, use `build/android/armeabi-v7a/lite_demo` and the corresponding 32-bit library path. Replace the `.adla` filename with your actual generated model filename.

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
cd examples/lite_transformer_en_fr/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **Optional Override:** If your compiler has a different prefix name, for example `aarch64-linux-gnu` instead of `aarch64-none-linux-gnu`, you can override the default by setting the `GCC_COMPILER` variable:

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/linux/64/lite_demo`
* 32-bit: `build/linux/32/lite_demo`

#### Mode 2: Yocto/Debian/Armbian Build

##### 1. Prerequisites

* Yocto SDK installed.
* CMake Toolchain file available.
* Prebuilt OpenCV located at `../../../dependency/opencv/` (relative to the script directory).

##### 2. Setup Environment

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. Build

```bash
cd examples/lite_transformer_en_fr/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **Note:** You can also use the `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` environment variables instead of passing the `-s` and `-t` flags.

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/yocto/64/lite_demo`
* 32-bit: `build/yocto/32/lite_demo`

#### 3. Example Run

The following example uses the default 64-bit Linux build.

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/lite_demo /data/local/tmp/
adb push ../model/lite_transformer_en_fr_w8a16.adla /data/local/tmp/
adb push ../assets/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x lite_demo

# Usage: ./lite_demo <model.adla> <en-fr_text_assets> [--max-new-tokens N] <texts ...>
./lite_demo lite_transformer_en_fr_w8a16.adla assets/ --max-new-tokens 64 "This is a translation test."
```

> **Note:** Replace the `.adla` filename with your actual generated model filename. Adjust the executable path if using a 32-bit or Yocto build.

## 5. Results

### Performance Feedback

By setting the log level to `INFO`, the program provides runtime performance information after inference. The console output may include:

* **Hardware Information:** System and ADLA library versions.
* **Model Overview:** Input and output tensor configurations.
* **NPU Metrics:** Inference latency and DRAM bandwidth usage.

### Translation Output

For each supplied English sentence, the demo performs tokenization, BPE encoding, Lite Transformer inference, BPE decoding, and text detokenization before printing the translated French sentence.

Example:

```text
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

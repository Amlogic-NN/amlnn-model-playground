[EN](README.md) | **CN**

# Lite Transformer 示例

本示例使用 AMLNN 运行 Lite Transformer 英译法翻译。完整流程如下：

1. 准备或下载 ONNX 模型和文本处理资源。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看翻译结果。

## 目录结构

```bash
examples/lite_transformer_en_fr/
├── assets/            # Vocabulary dictionaries and BPE codes
├── cpp/               # C++ demo and build scripts
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例使用的 Lite Transformer 模型架构、预训练模型资源、词表和 BPE codes 均来自 Lite Transformer 和 Fairseq 项目。Lite Transformer 软件采用 BSD 3-Clause License 发布，Fairseq 及其适用的模型资源采用 MIT License 发布。

本示例使用的转换后 ONNX 模型和量化 ADLA Lite Transformer 模型均源自这些第三方资源。重新分发模型或随附的文本处理资源时，请保留适用的版权、许可证、免责声明和署名声明。

完整的第三方许可证文本和署名声明请参阅 [THIRD_PARTY_NOTICES.txt](./THIRD_PARTY_NOTICES.txt)。

本示例中的模型预处理和后处理代码部分由 AI 辅助生成，随后经审阅并针对本项目进行了调整。如果其中任何部分无意中与现有作品相似并引起疑虑，请联系我们，我们将及时删除或调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载已准备好的 Lite Transformer 英译法 ONNX 模型：

- [在此下载 Lite Transformer ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/Lite_transformer/lite_transformer_en_fr.onnx)

下载所需的词表和 BPE 资源：

- [在此下载 Lite Transformer assets.zip！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/Lite_transformer/assets.zip)

将下载的 ONNX 模型放到 `examples/lite_transformer_en_fr/model/` 下，并将资源解压到 `examples/lite_transformer_en_fr/assets/` 下。

预期路径：

```text
examples/lite_transformer_en_fr/model/lite_transformer_en_fr.onnx
examples/lite_transformer_en_fr/assets/
├── dict.en.txt
├── dict.fr.txt
└── bpecodes
```

## 2. 将 ONNX 转换为 ADLA

从 `examples/lite_transformer_en_fr/py` 目录运行 ADLA 导出脚本：

```bash
cd examples/lite_transformer_en_fr/py
python export_adla.py \
  --onnx ../model/lite_transformer_en_fr.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | Lite Transformer `.onnx` 模型的路径。                                                           |
| `--target-platform` | 目标平台 ID。受支持平台的完整列表见[**此处**](../../docs/mapping.md)。                          |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                      |

转换完成后，AMLNN 生成的文件名将保持不变。

预期的模型路径：

```text
examples/lite_transformer_en_fr/model/lite_transformer_en_fr_w8a16.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需软件包：`amlnn`、`subword-nmt`、`sacremoses`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl subword-nmt sacremoses
```

### 在设备上运行

```bash
python lite_inference.py \
    --adla ../model/lite_transformer_en_fr_w8a16.adla \
    --assets ../assets \
    --texts "Hello world." "This is a translation test."
```

参数说明：

| 参数       | 说明                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------- |
| `--adla`   | Lite Transformer 英译法 `.adla` 模型的路径。                                                |
| `--assets` | 包含 `dict.en.txt`、`dict.fr.txt` 和 `bpecodes` 的目录。                                    |
| `--texts`  | 一个或多个待翻译的英文句子。请用引号将每个完整句子括起来。                                  |

脚本会逐句处理提供的英文句子，并将对应的法语翻译打印到控制台。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 已在系统中安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 预编译的 Android 版 OpenCV 位于 `dependency/opencv/` 文件夹中。

#### 1. 配置环境

导出 NDK 和 AMLNN 的路径，以便构建脚本找到所需的工具链和神经网络依赖。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

```bash
cd examples/lite_transformer_en_fr/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/lite_demo`
* 32 位：`build/android/armeabi-v7a/lite_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

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

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/lite_demo` 及对应的 32 位库路径。请将 `.adla` 文件名替换为你实际生成的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 文件夹必须添加到系统的 `PATH` 中。
* 预编译的 OpenCV 位于 `dependency/opencv/` 文件夹中。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 配置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds, add the 64-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# OR for 32-bit (arm) builds, add the 32-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

##### 2. 构建

```bash
cd examples/lite_transformer_en_fr/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名称不同，例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`，可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/lite_demo`
* 32 位：`build/linux/32/lite_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* CMake 工具链文件可用。
* 预编译的 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。

##### 2. 配置环境

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/lite_transformer_en_fr/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/lite_demo`
* 32 位：`build/yocto/32/lite_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

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

> **注意：** 请将 `.adla` 文件名替换为你实际生成的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包含：

* **硬件信息：** 系统版本和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延和 DRAM 带宽占用。

### 翻译输出

对于每个提供的英文句子，示例会依次执行分词、BPE 编码、Lite Transformer 推理、BPE 解码和文本反分词，然后打印翻译后的法语句子。

示例：

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

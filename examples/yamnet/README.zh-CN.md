[EN](README.md) | **CN**

# YAMNet 示例

本示例使用 AMLNN 运行 YAMNet 音频分类。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看音频分类结果。

## 目录结构

```bash
examples/yamnet/
├── cpp/               # C++ demo and build scripts
├── input/             # Input WAV audio files and class labels
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码基于 Apache License 2.0 授权。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

YAMNet 的架构与预训练权重最初来自官方 [TensorFlow YAMNet 仓库](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)，该仓库采用 **Apache License 2.0** 许可证。

允许再分发该编译后的 `.onnx` 二进制文件，这完全符合原始 Apache 2.0 许可证条款。`.onnx` 模型文件可在下方下载。下载该模型用于进一步商业使用或再分发的用户应保留此署名信息。

附带的 CSV 标签文件包含标准的 521 个 AudioSet 类别名称，直接来自原始 GitHub 仓库，此处提供以方便使用。

本示例中的代码部分由 AI 生成。如果其中任何部分与现有工作相似并引起疑虑，请联系我们，我们将予以处理。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载已准备好的 ONNX 模型：

### [在此下载 YAMNet ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/yamnet/yamnet.onnx)

将下载的模型放到以下路径：

```text
examples/yamnet/model/yamnet.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/yamnet/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/yamnet/py
python export_adla.py \
  --onnx ../model/yamnet.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | 输入 `.onnx` 模型的路径。                                                                       |
| `--target-platform` | 目标平台 ID。完整的支持平台列表见 [**此处**](../../docs/mapping.md)。                           |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                      |

转换后会保留 AMLNN 生成的文件名。对于当前的 `w8a16` 搭配 `activation_dtype="f16"` 配置，预期的模型路径为：

```text
examples/yamnet/model/yamnet_w8a16.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需软件包：`numpy`、`amlnn`、`librosa`

### 安装依赖

```bash
pip install numpy amlnn_edge_toolkit-1.0.0-cp310-cp310-linux_aarch64.whl librosa
```

### 在设备上运行

```bash
python yamnet_inference.py \
    --adla ../model/yamnet_w8a16.adla \
    --labels ../input/yamnet_class_map.csv \
    --audio-dir ../input
```

参数说明：

| 参数          | 说明                                          |
| ------------- | --------------------------------------------- |
| `--adla`      | YAMNet `.adla` 模型的路径。                   |
| `--labels`    | YAMNet 类别映射 `.csv` 文件的路径。           |
| `--audio-dir` | 包含 `.wav` 测试音频文件的目录。              |

该脚本会处理指定目录中的 `.wav` 音频文件，并为每个文件打印 Top-5 音频分类结果。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。

#### 1. 配置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/yamnet/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/yamnet_demo`
* 32 位：`build/android/armeabi-v7a/yamnet_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/yamnet_demo /data/local/tmp/
adb push ../model/yamnet_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x yamnet_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./yamnet_demo <model_path> <audio_dir> <labels.csv>
./yamnet_demo yamnet_w8a16.adla input/ input/yamnet_class_map.csv
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/yamnet_demo` 及对应的 32 位库路径。请将 `yamnet_w8a16.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 目录必须已添加到系统的 `PATH` 中。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 配置环境

将下载的工具链添加到 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本能够找到编译器和神经网络依赖。

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
cd examples/yamnet/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果编译器的前缀名称不同，例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`，可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/yamnet_demo`
* 32 位：`build/linux/32/yamnet_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 前置条件

* 已安装 Yocto SDK。
* 可用的 CMake Toolchain 文件。

##### 1. 配置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 2. 构建

```bash
cd examples/yamnet/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 选项。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/yamnet_demo`
* 32 位：`build/yocto/32/yamnet_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/yamnet_demo /data/local/tmp/
adb push ../model/yamnet_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x yamnet_demo

# Usage: ./yamnet_demo <model_path> <audio_dir> <labels.csv>
./yamnet_demo yamnet_w8a16.adla input/ input/yamnet_class_map.csv
```

> **注意：** 请将 `yamnet_w8a16.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在执行完成时提供实时的性能指标。控制台日志将显示关键的硬件与执行细节，包括：

* **硬件信息：** 系统与 ADLA 库版本。
* **模型概览：** 基本的输入/输出配置。
* **NPU 指标：** 总推理时间（时延）和总 DRAM 带宽消耗。

### 分类输出

对于每个音频文件，程序会打印前 5 个分类结果及其对应分数：

```text
============================================================
Processing audio: "drum.wav"
============================================================
  Resampling from 44100Hz to 16000Hz...
Total Inference time (all frames): 10.3505 ms
  1. Music                          (0.8675)
  2. Percussion                     (0.6694)
  3. Musical instrument             (0.5879)
  4. Drum                           (0.5751)
  5. Snare drum                     (0.4735)
============================================================
```

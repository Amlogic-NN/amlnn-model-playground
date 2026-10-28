[EN](README.md) | **CN**

# Whisper 示例

本示例使用 AMLNN 运行 Whisper。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看音频转写结果。

## 目录结构

```bash
examples/whisper/
├── cpp/               # C++ demo and build scripts
├── input/             # Input WAV audio files for demo
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码基于 Apache License 2.0 授权。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

随附的 Whisper 编码器和解码器 ONNX 模型是由多语言 OpenAI Whisper checkpoint 转换得到的产物。OpenAI 最初开发了 Whisper 架构和模型权重，并以 MIT License 发布了上游 Whisper 实现和模型权重。

用于生成这些转换后 ONNX 模型的 checkpoint 通过 ModelScope 获取。

在遵守适用的上游许可证条款的前提下，允许再分发和商业使用转换后的 ONNX 模型文件。再分发这些模型文件或相关 Whisper 实现的实质性部分时，必须保留 OpenAI 版权声明和 MIT License。

> MIT License
>
> Copyright (c) 2022 OpenAI
>
> 特此免费授予任何人获取本软件及相关文档文件（以下称“软件”）副本的权利，并允许在不受限制的情况下处理本软件，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售本软件副本的权利，以及允许获得本软件的人这样做，但须满足以下条件：
>
> 上述版权声明和本许可声明应包含在本软件的所有副本或主要部分中。
>
> 本软件按“原样”提供，不附带任何明示或暗示的保证，包括但不限于对适销性、特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任承担责任，无论是在合同诉讼、侵权行为还是其他方面，由本软件或本软件的使用或其他交易引起或与之相关。

编码器和解码器计算图已转换为 ONNX，并且可能为兼容目标推理运行时做过修改。这些转换后的模型文件并非 OpenAI 官方发布，OpenAI 也未对本项目背书。

随附的 `tokenizer_info.bin` 和 `data.bin` 文件由对应的 Whisper tokenizer 和特征提取器文件生成，按相同的上游署名和适用的许可证条款分发。

本示例中的模型预处理和后处理代码部分由 AI 生成，其中包含源自上游 Whisper 实现或与其兼容的逻辑。AI 辅助不会替代或改变任何上游代码、模型、tokenizer 数据或其他第三方资产的许可证要求。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载已准备好的 Whisper 编码器和解码器 ONNX 模型以及 tokenizer 文件：

- [点此下载 Whisper 编码器 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/whisper/whisper_encoder_static_sim.onnx)
- [点此下载 Whisper 解码器 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/whisper/whisper_decoder_static_sim.onnx)
- [点此下载 Whisper Tokenizer 压缩包！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/whisper/whisper_tokenizer.zip)

将下载的 ONNX 模型放到以下路径：

```text
examples/whisper/model/whisper_encoder_static_sim.onnx
examples/whisper/model/whisper_decoder_static_sim.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/whisper/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/whisper/py
python export_adla.py \
  --encoder-onnx ../model/whisper_encoder_static_sim.onnx \
  --decoder-onnx ../model/whisper_decoder_static_sim.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--encoder-onnx`    | Whisper 编码器 `.onnx` 模型的路径。                                                             |
| `--decoder-onnx`    | Whisper 解码器 `.onnx` 模型的路径。                                                             |
| `--target-platform` | 目标平台 ID。支持的平台完整列表见 [**此处**](../../docs/mapping.md)。                          |
| `--output-dir`      | （可选）生成的 `.adla` 模型保存目录，默认为 `../model`。                                        |

转换后会保留 AMLNN 生成的原始文件名。在当前 `w8a16` 且 `activation_dtype="f16"` 的配置下，预期模型路径为：

```text
examples/whisper/model/whisper_encoder_static_sim_w8a16.adla
examples/whisper/model/whisper_decoder_static_sim_w8a16.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 依赖包：`amlnn`、`librosa`、`transformers`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl librosa transformers
```

### 在设备上运行

```bash
python whisper_inference.py \
    --enc ../model/whisper_encoder_static_sim_w8a16.adla \
    --dec ../model/whisper_decoder_static_sim_w8a16.adla \
    --tokenizer ../tokenizer/ \
    --audio-file ../input/jfk.wav
```

参数说明：

| 参数                   | 说明                                       |
| ---------------------- | ------------------------------------------ |
| `--enc` | Whisper 编码器 `.adla` 模型的路径。        |
| `--dec` | Whisper 解码器 `.adla` 模型的路径。        |
| `--tokenizer`      | Whisper tokenizer 目录的路径。             |
| `--audio-file`         | 输入的 `.wav` 音频文件路径。               |

该脚本会处理指定的 `.wav` 音频文件，并将转写结果打印到控制台。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 已安装 **Android NDK**（推荐 r25e）。
* 已下载并解压 **AMLNN Toolkit**。
* 已设置 `ANDROID_NDK_PATH` 环境变量。

#### 1. 设置环境

导出 NDK 和 AMLNN 的路径，以便构建脚本能找到所需的工具链和神经网络依赖。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25e
export AMLNN_HOME=/path/to/amlnn-toolkit
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/whisper/cpp

# Build for 64-bit (arm64-v8a)
./build-android.sh -a arm64-v8a
```

可执行文件将生成在：

```text
build/android/arm64-v8a/whisper_demo
```

#### 3. 运行示例

以下示例使用 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/whisper_demo /data/local/tmp/
adb push ../model/whisper_encoder_static_sim_w8a16.adla /data/local/tmp/
adb push ../model/whisper_decoder_static_sim_w8a16.adla /data/local/tmp/
adb push ../cpp/data_bin /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x whisper_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./whisper_demo <encoder.adla> <decoder.adla> <audio.wav> <data_bin_dir>
# Set GET_TIME=1 before the command to print preprocessing, encoder, decoder, and total inference times.
# Set GET_OUTPUTS_SIZE=1 before the command to print the number of decoder inference calls.
GET_TIME=1 GET_OUTPUTS_SIZE=1 ./whisper_demo whisper_encoder_static_sim_w8a16.adla whisper_decoder_static_sim_w8a16.adla input/jfk.wav data_bin
```

> **注意：** 将 ADLA 文件名替换为你实际生成的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 方式一：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 目录必须加入系统的 `PATH`。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

将下载的工具链加入 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本能找到编译器和神经网络依赖。

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit

# For 64-bit (aarch64) builds, add the 64-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# OR for 32-bit (arm) builds, add the 32-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

##### 2. 构建

```bash
cd examples/whisper/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名称不同，例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`，可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/whisper_demo`
* 32 位：`build/linux/32/whisper_demo`

#### 方式二：Yocto 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake Toolchain 文件。
* 预编译的 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit
```

##### 3. 构建

```bash
cd examples/whisper/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不用传递 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/whisper_demo`
* 32 位：`build/yocto/32/whisper_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/whisper_demo /data/local/tmp/
adb push ../model/whisper_encoder_static_sim_w8a16.adla /data/local/tmp/
adb push ../model/whisper_decoder_static_sim_w8a16.adla /data/local/tmp/
adb push ../cpp/data_bin /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x whisper_demo

# Usage: ./whisper_demo <encoder.adla> <decoder.adla> <audio.wav> <data_bin_dir>
# Set GET_TIME=1 before the command to print preprocessing, encoder, decoder, and total inference times.
# Set GET_OUTPUTS_SIZE=1 before the command to print the number of decoder inference calls.
GET_TIME=1 GET_OUTPUTS_SIZE=1 ./whisper_demo whisper_encoder_static_sim_w8a16.adla whisper_decoder_static_sim_w8a16.adla input/jfk.wav data_bin
```

> **注意：** 将 ADLA 文件名替换为你实际生成的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 音频转写输出

对于每个音频文件，程序会将音频切分为多个片段，并打印拼接后的转写结果。

输出示例：

```text
============================================================
Processing audio: ../input/jfk.wav
Segments: 1
============================================================
Processing segment [1/1]...
Segment transcription: And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
============================================================
Audio Text:
And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
============================================================
pre-process time             : 36ms
encoder inference time[0]    : 700ms
============================================================
decoder inference time[0]    : 51ms
decoder inference time[1]    : 50ms
decoder inference time[2]    : 39ms
decoder inference time[3]    : 37ms
decoder inference time[4]    : 40ms
decoder inference time[5]    : 39ms
decoder inference time[6]    : 42ms
decoder inference time[7]    : 40ms
decoder inference time[8]    : 43ms
decoder inference time[9]    : 43ms
decoder inference time[10]   : 43ms
decoder inference time[11]   : 44ms
decoder inference time[12]   : 43ms
decoder inference time[13]   : 43ms
decoder inference time[14]   : 43ms
decoder inference time[15]   : 43ms
decoder inference time[16]   : 43ms
decoder inference time[17]   : 43ms
decoder inference time[18]   : 43ms
decoder inference time[19]   : 43ms
decoder inference time[20]   : 43ms
decoder inference time[21]   : 43ms
decoder inference time[22]   : 43ms
decoder inference time[23]   : 51ms
============================================================
model->whisper encoder avg   : 700ms
model->whisper decoder avg   : 43ms
model->whisper total time    : 1771ms
```

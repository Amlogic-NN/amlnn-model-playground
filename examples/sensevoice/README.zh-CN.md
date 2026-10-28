[EN](README.md) | **CN**

# SenseVoice 示例

本示例使用 AMLNN 运行 SenseVoice 多语种语音识别。完整流程如下：

1. 准备或导出 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 或 C++ 示例。
4. 查看识别结果和性能分析报告。

## 目录结构

```text
examples/sensevoice/
|-- cpp/                  # C++ demo and build scripts
|-- input/                # Input wav files for demo (en/zh/ja/ko/yue)
|-- model/                # Put ONNX and ADLA models here
|-- py/                   # Python conversion and demo scripts
`-- README.md
```



## 许可证

本示例代码基于 Apache License 2.0 许可发布。详见仓库根目录的 [LICENSE](../../LICENSE) 文件。

SenseVoice 模型由第三方（FunAudioLLM）提供。在再分发或商业使用前，请查阅并遵守模型来源的许可证与使用条款。

本示例中的预处理和后处理代码参考了公开的 kaldi-native-fbank 与 SenseVoice 实现。如其中部分内容与已有工作相似并引起疑虑，请联系我们，我们将予以删除或调整。

## 1. 准备 ONNX 模型

模型来源：

[https://github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)

上游 SenseVoice 仓库提供的 `export.py` 会生成 **4 输入的 ONNX**，显式包含 `speech`、`speech_lengths`、`language`、`textnorm` 张量。该 4 输入 ONNX **不能**直接在 ADLA 上使用——因此本示例提供了一个改造后的导出脚本，把音频长度（`speech_lengths`）固定为计算图常量并从输入列表中移除，从而得到 **3 输入的 ONNX** 变体，显式包含 `x`、`language`、`text_norm` 张量，可被 ADLA 工具链处理。

### 导出 ONNX

安装或克隆 SenseVoice 模型包：

```bash
git clone https://github.com/FunAudioLLM/SenseVoice.git
```

将本示例中改造后的 3 输入导出脚本复制到 SenseVoice 仓库，然后用它替代上游的 `export.py` 运行：

```bash
cd SenseVoice
pip install -r requirements.txt
pip install onnx onnxruntime onnxscript
cp /path/to/amlnn-model-playground/examples/sensevoice/py/export-onnx-3input.py .
python export-onnx-3input.py
```

脚本会生成 `onnx-3input/` 目录，其中包含 FP32 3 输入 ONNX 模型、动态量化变体以及 `tokens.txt`。首次运行时会从 ModelScope 下载 `iic/SenseVoiceSmall` 权重（需要联网）。

将生成的 `onnx-3input/sensevoice_small.onnx` 和 `onnx-3input/tokens.txt` 放到 `examples/sensevoice/model/` 目录下，供下一步使用：

```text
examples/sensevoice/model/sensevoice_small.onnx
examples/sensevoice/model/tokens.txt
```



## 2. 将 ONNX 转换为 ADLA

在 `examples/sensevoice/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/sensevoice/py
python export_adla.py \
  --onnx ../model/sensevoice_small.onnx \
  --target-platform 007 \
  --adla ../model/SenseVoice_w8a16_A311Y3.adla
```

参数：


| 参数                | 说明                                                             |
| ------------------- | ---------------------------------------------------------------- |
| `--onnx`            | SenseVoice 3 输入 FP32 ONNX 模型的路径。                         |
| `--target-platform` | 平台 ID。A311D2：003。S905X5：005。C302X2：006。A311Y3：007。    |
| `--adla`            | 可选的 ADLA 输出路径（默认：`../model`）。                       |


> 此处未提供校准数据集；`compile()` 使用随机校准数据运行。精度可能下降——正式使用时请准备 NPY 校准数据集（建议 100-300 组，覆盖 `x`、`language`、`text_norm` 全部三个输入）。

转换完成后，预期的模型路径为：

```text
examples/sensevoice/model/SenseVoice_w8a16_A311Y3.adla
```



## 3. 运行 Python 示例



### 前置条件

- Python 3.10
- `numpy`
- `kaldi-native-fbank`
- 适用于目标设备的 AMLNN Python wheel

在目标设备上安装依赖：

```bash
pip install numpy kaldi-native-fbank amlnn_edge_toolkit-1.0.0-cp310-cp310-linux_aarch64.whl
```

运行推理：

```bash
cd examples/sensevoice/py
python sensevoice_inference.py \
  --model ../model/SenseVoice_w8a16_A311Y3.adla \
  --tokens ../model/tokens.txt \
  --wav ../input/en.wav \
  --lang en
```

Python 示例参数：


| 参数       | 说明                                                        |
| ---------- | ----------------------------------------------------------- |
| `--model`  | ADLA 模型的路径。                                           |
| `--tokens` | `tokens.txt` 的路径。                                       |
| `--wav`    | PCM16 WAV 输入文件（建议 16 kHz 单声道）。                  |
| `--lang`   | `auto` / `zh` / `en` / `ja` / `ko` / `yue`。                |
| `--itn`    | 逆文本归一化：`0`（关闭，默认）或 `1`（开启）。             |




## 4. 运行 C++ 示例



### 为 Android 构建

前置条件：

- Android NDK，推荐 r25e
- 已设置 `ANDROID_NDK_PATH`、`ANDROID_NDK` 或 `ANDROID_NDK_HOME`
- `AMLNN_HOME` 指向 amlnn_runtime
- `kaldi-native-fbank`（由 CMake 自动获取）

构建：

```bash
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime ./build-android.sh -a arm64-v8a
```

生成的可执行文件位于：

```text
examples/sensevoice/cpp/build/android/sensevoice_demo
```



### 为 Linux 构建

Linux 构建支持两种方式：**标准交叉编译**（默认）和 **Yocto SDK 编译**。

#### 方式 1：标准 Linux 交叉编译（默认）

**前置条件：**

- GCC 交叉编译工具链（推荐 GCC 10.3+）
- 已将工具链的 `bin/` 目录加入 `PATH`
- 已设置 `AMLNN_HOME` 环境变量
- `kaldi-native-fbank`（由 CMake 自动获取）

**1. 配置环境：**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# For 32-bit (arm) builds:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

**2. 构建：**

```bash
cd examples/sensevoice/cpp

# Build for 64-bit (default)
./build-linux.sh

# Or explicitly:
./build-linux.sh -b 64
./build-linux.sh -a aarch64

# Build for 32-bit
./build-linux.sh -b 32
./build-linux.sh -a armhf
```

如果编译器前缀不同，可通过 `GCC_COMPILER` 覆盖：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

构建产物：

```text
build/linux/64/sensevoice_demo
build/linux/32/sensevoice_demo
```



#### 方式 2：Yocto/Debian/Armbian 构建

**前置条件：**

- 已安装 Yocto SDK
- 已有可用的 CMake 工具链文件

**构建：**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/sensevoice/cpp

# Build for Yocto 64-bit (default)
./build-linux.sh -m yocto -b 64 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

也可以通过设置 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 来替代传入 `-s` 和 `-t`。

构建产物：

```text
build/yocto/64/sensevoice_demo
build/yocto/32/sensevoice_demo
```



### 在设备上运行



#### Android

推送文件：

```bash
adb push build/android/sensevoice_demo /data/local/tmp/
adb push ../model/SenseVoice_w8a16_A311Y3.adla /data/local/tmp/
adb push ../model/tokens.txt /data/local/tmp/
adb push ../input/en.wav /data/local/tmp/
```

运行：

```bash
adb shell
cd /data/local/tmp
chmod +x sensevoice_demo
export LD_LIBRARY_PATH=/vendor/lib64

./sensevoice_demo \
  --model SenseVoice_w8a16_A311Y3.adla \
  --tokens tokens.txt \
  --lang en \
  --wav en.wav
```



#### Linux

将可执行文件、模型、tokens 和 wav 文件复制到目标设备，然后运行：

```bash
./sensevoice_demo \
  --model SenseVoice_w8a16_A311Y3.adla \
  --tokens tokens.txt \
  --lang en \
  --wav en.wav
```

用法：

```text
./sensevoice_demo --model <model.adla> --tokens <tokens.txt> --lang <auto|zh|en|ja|ko|yue> --wav <input.wav> [--itn 0|1]
```



## 5. 结果



### 识别输出

示例会打印识别标签和转写文本：

```text
language: <|en|>
emotion:  <|NEUTRAL|>
event:    <|Speech|>
itn:      <|woitn|>
text:     ...
```



### 测试音频

`input/` 目录下的示例 wav 文件：


| 文件      | 语言      |
| --------- | --------- |
| `en.wav`  | 英语      |
| `zh.wav`  | 中文      |
| `ja.wav`  | 日语      |
| `ko.wav`  | 韩语      |
| `yue.wav` | 粤语      |




### 集成

如需将引擎集成到自己的应用中，请参阅 C API 头文件：`cpp/src/sense_voice.h`。

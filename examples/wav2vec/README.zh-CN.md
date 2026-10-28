[EN](README.md) | **CN**

# Wav2Vec2 示例

本示例使用 AMLNN 运行 Wav2Vec2 自动语音识别。完整流程如下：

1. 准备 ONNX 模型，或下载预先准备好的 ADLA 模型。
2. 使用该 ADLA 模型运行 Python 示例。
3. 使用该 ADLA 模型运行 C++（Linux/Android）示例。
4. 查看音频转写结果。

## 目录结构

```bash
examples/wav2vec/
├── cpp/               # C++ demo and build scripts
├── input/             # Input WAV audio files for demo
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例使用的 Wav2Vec2 模型基于 Facebook AI 发布的 [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) checkpoint。该 checkpoint 以 **Apache License 2.0** 许可发布，并使用 960 小时 16 kHz LibriSpeech 音频针对英语自动语音识别进行了预训练和微调。

原始的 Wav2Vec 2.0 模型实现通过 [fairseq 仓库](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) 发布，该仓库以 MIT License 许可分发。

本示例附带的 ONNX 模型是原始预训练 checkpoint 的转换版本。在遵循其条款的前提下，Apache License 2.0 允许以源码或编译形式使用、修改和再分发。再分发该 ONNX 模型时，请保留适用的版权与署名声明，附上 Apache License 2.0 的副本，标明所做的修改，并保留原始作品随附的任何适用的 `NOTICE` 信息。

本示例使用的 32-token CTC 词表来自原始 `facebook/wav2vec2-base-960h` checkpoint 随附的 `vocab.json` 文件。本示例不包含也不再分发 LibriSpeech 数据集。任何另行获取的音频或数据集文件仍受其各自的许可证条款约束。

本示例中的部分模型预处理和后处理代码在 AI 辅助下开发。如发现任何疑似许可证或署名问题，请反馈给我们，以便核查并更正。

## 1. 准备 ADLA 模型

### 下载 ADLA

下载预先准备好的 ADLA 模型：

- [在此下载 Wav2Vec2 ADLA 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/wav2vec2/wav2vec2_base_960h_20s.adla)

将下载的模型放到以下路径：

```text
examples/wav2vec/model/wav2vec2_base_960h_20s.adla
```

<!-- ## 2. 将 ONNX 转换为 ADLA

如果要自行转换 ONNX 模型而不使用预先准备好的 ADLA 模型，请在 `examples/wav2vec/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/wav2vec/py
python export_adla.py \
  --onnx ../model/wav2vec2_base_960h_20s.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | 输入 `.onnx` 模型的路径。                                                                       |
| `--target-platform` | 目标平台 ID。完整的目标平台列表见 [**此处**](../../docs/mapping.md)。                           |
| `--output-dir`      | （可选）生成的 `.adla` 模型保存目录。默认为 `../model`。                                        |

转换后会保留 AMLNN 生成的文件名。例如，对于使用当前 `w16a16` 配置的 `wav2vec.onnx`，预期的模型路径为：

```text
examples/wav2vec/model/wav2vec_w16a16.adla
```

当前转换配置使用 `w16a16` 和 `activation_dtype="f16"`，因此无需指定量化数据集参数。 -->

## 2. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需依赖包：`amlnn`、`librosa`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl librosa
```

### 在设备上运行

```bash
python wav2vec_inference.py \
    --adla ../model/wav2vec2_base_960h_20s.adla \
    --audio-dir ../input
```

参数说明：

| 参数           | 说明                                     |
| -------------- | ---------------------------------------- |
| `--adla` | Wav2Vec2 `.adla` 模型的路径。            |
| `--audio-dir`  | 包含 `.wav` 音频文件的目录。             |

脚本会自动处理指定音频目录中的 `.wav` 音频文件。音频会按模型要求切分为多个片段，拼接后的转写结果将打印到控制台。

## 3. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。

#### 1. 设置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/wav2vec/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/wav2vec_demo`
* 32 位：`build/android/armeabi-v7a/wav2vec_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/wav2vec_demo /data/local/tmp/
adb push ../model/wav2vec2_base_960h_20s.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x wav2vec_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./wav2vec_demo <model_path> <audio_dir>
./wav2vec_demo wav2vec2_base_960h_20s.adla input/
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/wav2vec_demo` 及对应的 32 位库路径。请将 `wav2vec2_base_960h_20s.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 文件夹必须添加到系统的 `PATH` 中。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

将下载的工具链添加到 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本找到编译器和神经网络依赖。

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
cd examples/wav2vec/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名不同（例如使用 `aarch64-linux-gnu` 而不是 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/wav2vec_demo`
* 32 位：`build/linux/32/wav2vec_demo`

#### 模式 2：Yocto 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* CMake 工具链文件可用。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/wav2vec/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不必传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/wav2vec_demo`
* 32 位：`build/yocto/32/wav2vec_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/wav2vec_demo /data/local/tmp/
adb push ../model/wav2vec2_base_960h_20s.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x wav2vec_demo

# Usage: ./wav2vec_demo <model_path> <audio_dir>
./wav2vec_demo wav2vec2_base_960h_20s.adla input/
```

> **注意：** 请将 `wav2vec2_base_960h_20s.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 4. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理完成后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 音频转写输出

对于每个 `.wav` 音频文件，程序会按模型要求将音频切分为多个片段，逐段处理，并将拼接后的转写结果打印到控制台。

输出示例：

```text
============================================================
Processing [1/2]: Alexanders_Bridge_Chapter_II_by_Willa_Cather.wav
============================================================
Segments: 8
Processing segment [1/8]...
Processing segment [2/8]...
Processing segment [3/8]...
Processing segment [4/8]...
Processing segment [5/8]...
Processing segment [6/8]...
Processing segment [7/8]...
Processing segment [8/8]...
Transcription: MAIN HALL LIKED ALEXANDER BECAUSE HE WAS AN ENGINEER HE HAD PRECONCEIVED IDEAS ABOUT EVERYTHING AND HIS IDEA ABOUT AMERICANS WAS THAT THEY SHOULD BE ENGINEERS ON MECHANICS IT'S TREMENDOUSLY WELL PUT ON TOO IT'S BEEN ON ONLY TWO WEEKS AND I'VE BEEN HALF A DOZEN TIMES ALREADY DO YOU KNOW ALEXANDER MAIN HLL LOOKED WITH PERPLEXITY UP INTO THE TOP OF THE HANSOM AND RUBBED HIS PINK CHEEK WITH HIS GLOVED FINGER DOYOU KNOW I SOMETIMES THINK OF TAKING TO CRITICISM SERIOUSLY MYSELF SHE SAVES HER HAND TOO SHE'S AT HER BEST IN THE SECONDACT HE'S BEEN WANTING TO MARRY HILDA THESE THREE YEARS AND MORE SHE DOESN'T TAKE UP WITH ANYBODY YOU KNOW IREN BERGOING ONE OF HER FAMILY TOLD ME IN CONFIDENCE THAT THERE WAS A ROMANCE SOMEWHERE BACK IN THE BEGINNING MAINHALL VOUCHED FOR HER CONSTANCY WITH A LOFTINESS THAT MADE ALEXANDER SMILE EVEN WHILE A KIND OF RAPID EXCITEMENT WAS TINGLING THROUGH HIM HE'S ANOTHER WHO'S AWFULLY KEEN ABOUT HER LET ME INTRODUCE YOU SIR HARRY TOWN MISTER BARTLEY ALEXANDER THE AMERICAN ENGINEER I SAY SIR HARRY THE LITTLE GIRL'S GOING FAMOUSLY TO NIGHT ISN'T SHE YOU KNOW I THOUGHT THE DANCE AF BI CONSCIENCE TO NIGHT FOR THE FIRST TIME WESTMERE AND I WERE BACK AFTER THE FIRST ACT AND WE THOUGHT SHE SEEMED QUITE UNCERTAIN OF HERSELF A LITTLE ATTACK OF NERVES POSSIBLY HE WAS BEGINNING TO FEEL KEEN INTEREST IN THE SLENDER BAREFOOT DONKEY GIRL WHO SLIPPED IN AND OUT OF THE PLAY SINGING LIKE SOME ONE WINDING THROUGH A HLLY FIELD ONE NIGHT WHEN HE AND WINTIFORD WERE SITTING TOGETHER ON THE BRIDGE HE TOLD HER THAT THINGS HAD HAPPENED WHILE HE WAS STUDYING ABROAD THAT HE WAS SORRY FOR ONE THING IN PARTICULAR AND HE ASKED HER WHETHER SHE THOUGHT SHE OUGHT TO KNOW ABOUT THEM SHE CONSIDERED FOR A MOMENT AND THEN SAID NO I THINK NOT THOUGH I AM GLAD YOU ASK ME AFTER THAT IT WAS EASY TO FORGET ACTUALLY TO FORGET OF COURSE HE REFLECTED SHE ALWAYS HAD THAT COMBINATION OF SOMETHING HOMELY AND SENSIBLE AND SOMETHING UTTERLY WILD AND DAFT SHE MUST CARE ABOUT THE THEATRE A GREAT DEAL MORE THAN SHE USED TO I'M GLAD SHE'S HELD HER OWN SINCE AFTER ALL WE WERE AWFULLY YOUNG I SHOULDN'T WONDER IF SHE COULD LAUGH ABOUT IT WITH ME NOW
============================================================
```

## 第三方声明

示例音频文件源自 LibriSpeech ASR 语料库，该语料库由 Vassil Panayotov、Guoguo Chen、Daniel Povey 和 Sanjeev Khudanpur 创建。

LibriSpeech 以 Creative Commons Attribution 4.0 International License（CC BY 4.0）许可分发。

来源：OpenSLR SLR12，LibriSpeech `test-clean`，说话人/章节 `4446/2271`。

原始文件：`4446-2271-0000.flac` 至 `4446-2271-0024.flac`。

修改说明：原始 LibriSpeech 语音样本经过拼接、从 FLAC 转换为 WAV 并重采样，用作示例输入。这并不意味着 LibriSpeech 作者或贡献者的认可。

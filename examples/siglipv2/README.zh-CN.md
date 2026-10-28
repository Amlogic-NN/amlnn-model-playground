[EN](README.md) | **CN**

# SigLIP2

本示例使用 AMLNN 运行 SigLIP2。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看图文匹配结果。

## 目录结构

```bash
examples/siglipv2/
├── cpp/                 # C++ demo and build scripts
├── input/               # Input images for demo
├── model/               # Put ONNX and ADLA models here
├── py/                  # Python conversion and demo scripts
└── tokenizer/           # SigLIP2 tokenizer directory
```

## 许可证

本示例代码使用 Apache License 2.0 许可证。详情请参见仓库根目录的 [LICENSE](../../LICENSE) 文件。

本项目分发 SigLIPv2 ONNX 模型，该模型最初由 Google 开发，取自 Hugging Face。模型权重与网络结构均以 Apache License 2.0 提供。如果你进一步再分发这些模型文件或将其用于商业用途，请确保保留相应的许可证声明。

本示例中的模型预处理和后处理代码部分由 AI 生成。若有任何部分与现有工作相似并引起疑虑，请联系我们，我们会将其删除或调整。

## 1. 准备 ONNX 模型

### 下载模型

下载已准备好的模型和 tokenizer：

### [文本编码器 ONNX 模型](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/siglipv2/siglip2_text_encoder_sim.onnx)
### [图像编码器 ONNX 模型](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/siglipv2/siglip2_vision_encoder_sim.onnx)
### [Siglipv2 Tokenizer 压缩包](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/siglipv2/siglip_tokenizer.zip)

将 ONNX 模型放到 `examples/siglipv2/model/` 下，并将 tokenizer 解压到 `examples/siglipv2/tokenizer/` 下。

预期路径：

```text
examples/siglipv2/model/siglip2_text_encoder_sim.onnx
examples/siglipv2/model/siglip2_vision_encoder_sim.onnx
examples/siglipv2/tokenizer/
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/siglipv2/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/siglipv2/py
python export_adla.py \
    --text-onnx ../model/siglip2_text_encoder_sim.onnx \
    --image-onnx ../model/siglip2_vision_encoder_sim.onnx \
    --target-platform 007 \
    --output-dir ../model
```

| 参数                   | 说明                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--text-onnx`          | 文本编码器 `.onnx` 模型的路径。                                                                          |
| `--image-onnx`         | 图像编码器 `.onnx` 模型的路径。                                                                          |
| `--text-dataset-path`  | （可选）`.txt` 文件的路径，其中包含用于文本模型量化校准的输入路径。                                     |
| `--image-dataset-path` | （可选）`.txt` 文件的路径，其中包含用于图像模型量化校准的输入路径。                                     |
| `--target-platform`    | 目标平台 ID。完整的目标平台支持列表见[**此处**](../../docs/mapping.md)。                                |
| `--output-dir`         | （可选）生成的 `.adla` 模型保存目录。默认为 `../model`。                                                 |

> **量化数据集说明**
>
> 使用 `activation_dtype="f16"` 时无需校准数据集，例如 `w8a16 (f16)` 或 `w16a16 (f16)`。
>
> 对于需要校准的量化模式，如果没有提供数据集，AMLNN 会生成随机校准数据而不会报错退出，这可能导致模型精度严重下降。

使用生成的模型时，预期路径大致如下：

```text
examples/siglipv2/model/siglip2_text_encoder_sim.adla
examples/siglipv2/model/siglip2_vision_encoder_sim.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需软件包：`numpy`、`opencv-python`、`amlnn`

### 安装依赖

```bash
pip install numpy opencv-python amlnn_edge_toolkit-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

SigLIP2 Python 示例通过 `--prompt` 接收候选标签，并应用 `--prompt-template` 指定的提示模板。

```bash
python siglipv2_inference.py \
    --vision ../model/siglip2_vision_encoder_sim.adla \
    --text ../model/siglip2_text_encoder_sim.adla \
    --tokenizer ../tokenizer/ \
    --image-dir ../input \
    --prompt "red bus" "blue car" "person" "dog"
```

参数说明：

| 参数                | 说明                                                                   |
| ------------------- | ---------------------------------------------------------------------- |
| `--vision`          | 视觉编码器 `.adla` 模型的路径。                                        |
| `--text`            | 文本编码器 `.adla` 模型的路径。                                        |
| `--tokenizer`   | 存放 SigLIP2 tokenizer 文件的目录。                                    |
| `--image-dir`       | 存放测试图像的目录。                                                   |
| `--prompt`          | 与每张图像比对的候选标签。                                             |
| `--prompt-template` | （可选）包含 `{}` 的提示模板。默认为 `This is a photo of {}.`。        |
| `--top-k`           | （可选）输出结果的数量。默认为 `5`。                                   |

该脚本会处理指定图像目录中的图像，并输出排序最高的图文匹配结果。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。

#### 1. 设置环境

导出 NDK 和 AMLNN 的路径，以便构建脚本能够找到所需的工具链和神经网络依赖。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/siglipv2/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在与所选 Android ABI 对应的 build 目录中。

* 64 位：`build/android/arm64-v8a/siglipv2_demo`
* 32 位：`build/android/armeabi-v7a/siglipv2_demo`

#### 3. 运行示例

C++ 可执行文件接收视觉模型、文本模型、tokenizer 目录、图像目录和候选文本。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/siglipv2_demo /data/local/tmp/
adb push ../model/siglip2_vision_encoder_sim.adla /data/local/tmp/
adb push ../model/siglip2_text_encoder_sim.adla /data/local/tmp/
adb push data_bin/ /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x siglipv2_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage:
# ./siglipv2_demo <vision_model.adla> <text_model.adla> <data_bin> <image_dir> [texts ...]
./siglipv2_demo \
    siglip2_vision_encoder_sim.adla \
    siglip2_text_encoder_sim.adla \
    ./data_bin \
    ../input/ \
    "a red handbag" \
    "a blue jacket" \
    "a red bus"
```
> **注意：** 如需生成 data_bin，`siglipv2/py/` 目录中提供了 `compile_tokenizer.py`。

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/siglipv2_demo` 以及对应的 32 位库路径。请将 `.adla` 模型文件名替换为你实际生成的模型文件名。

### 为 Linux 构建

Linux 构建过程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* 一个 GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 目录必须加入系统的 `PATH`。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

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
cd examples/siglipv2/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

可执行文件将生成在与所选架构对应的 build 目录中：

* 64 位：`build/linux/64/siglipv2_demo`
* 32 位：`build/linux/32/siglipv2_demo`

##### 可选覆盖

如果你的编译器使用不同的前缀名，可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 有可用的 CMake Toolchain 文件。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/siglipv2/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不必传入 `-s` 和 `-t` 参数。

可执行文件将生成在与所选架构对应的 build 目录中：

* 64 位：`build/yocto/64/siglipv2_demo`
* 32 位：`build/yocto/32/siglipv2_demo`

#### 运行示例

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/siglipv2_demo /data/local/tmp/
adb push ../model/siglip2_vision_encoder_sim.adla /data/local/tmp/
adb push ../model/siglip2_text_encoder_sim.adla /data/local/tmp/
adb push data_bin/ /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x siglipv2_demo

# Usage:
# ./siglipv2_demo <vision_model.adla> <text_model.adla> <data_bin> <image_dir> [texts ...]
./siglipv2_demo \
    siglip2_vision_encoder_sim.adla \
    siglip2_text_encoder_sim.adla \
    ./data_bin \
    ../input/ \
    "a red handbag" \
    "a blue jacket" \
    "a red bus"
```

> **注意：** 如需生成 data_bin，`siglipv2/py/` 目录中提供了 `compile_tokenizer.py`。

> **注意：** 请将 `.adla` 模型文件名替换为你实际生成的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 图文匹配输出

示例会对每张输入图像，将图像特征向量与提供的文本提示进行比对，并按分数从高到低输出匹配结果。

示例：

```text
============================================================
Processing image 1/4: apples.jpg
============================================================
    [1] apple: probability=0.162227, logit=-1.641751
    [2] a red handbag: probability=0.000022, logit=-10.720261
    [3] children: probability=0.000011, logit=-11.396953
    [4] dog: probability=0.000004, logit=-12.357706
    [5] elephant: probability=0.000001, logit=-13.892324
    Result saved to: siglip2_vision_encoder_sim_result/apples_result.txt
============================================================
```

结果展示了所提供提示的图文匹配分数，按匹配得分从高到低排序。

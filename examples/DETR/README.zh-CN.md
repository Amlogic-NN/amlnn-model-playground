[EN](README.md) | **CN**

# Detr 示例

本示例使用 AMLNN 运行 Detr。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看检测图像/结果。

## 目录结构

```bash
examples/DETR/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example detection output
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。详见仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例使用的 DETR 模型由第三方提供，源自 [Facebook Research DETR 项目](https://github.com/facebookresearch/detr)。原始 DETR 模型按 Apache License 2.0 分发。

本示例使用的转换并量化后的 DETR 模型由我方服务器按 Apache License 2.0 分发。再分发该模型时，请保留适用的版权、许可证和署名声明。

> Copyright 2020 - present, Facebook, Inc.
>
> 依据 Apache License, Version 2.0（以下简称“许可证”）授权；
> 除非遵守许可证，否则不得使用此文件。
> 许可证副本可从以下地址获取：
>
> ```
> http://www.apache.org/licenses/LICENSE-2.0
> ```
>
> 除非适用法律要求或书面同意，依据许可证分发的软件均按“原样”提供，
> 不附带任何明示或暗示的保证或条件。
> 有关许可证下具体权限与限制的条款，请参见许可证本身。

本示例中的模型预处理和后处理代码部分由 AI 辅助生成，随后经人工审阅并针对本项目进行了调整。若其中有内容无意间与既有工作相似并引起顾虑，请联系我们，我们会及时删除或调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载准备好的 ONNX 模型，放到 `examples/DETR/model/` 目录下：

- [点此下载 DETR ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/detr/detr.onnx)

预期路径：

```text
examples/DETR/model/detr.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/DETR/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/DETR/py
python export_adla.py \
  --onnx ../model/detr.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | 输入 `.onnx` 模型的路径。                                                                       |
| `--target-platform` | 目标平台 ID。完整的目标平台支持列表见[**此处**](../../docs/mapping.md)。                        |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                      |

转换完成后，生成的 AMLNN 文件名保持不变。在当前 `w16a16` 配置下，预期的模型路径为：

```text
examples/DETR/model/detr_w16a16.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需软件包：`amlnn`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

```bash
python detr_inference.py \
    --adla ../model/detr_w16a16.adla \
    --image-dir ../input
```

参数说明：

| 参数           | 说明                                                                                |
| -------------- | ----------------------------------------------------------------------------------- |
| `--adla` | 编译后的 `.adla` 格式 DETR 模型的路径。                                             |
| `--image-dir`  | 测试图像所在目录。                                                                  |
| `--conf`       | （可选）用于过滤检测结果的置信度阈值。默认为 `0.5`。                                |

脚本会自动处理指定图像目录下的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将结果保存到 `{model_name}_result` 文件夹。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 位于 `dependency/opencv/` 文件夹中的 Android 预编译 OpenCV。

#### 1. 设置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖库）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/DETR/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/detr_demo`
* 32 位：`build/android/armeabi-v7a/detr_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/detr_demo /data/local/tmp/
adb push ../model/detr_w16a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x detr_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./detr_demo <model_path> <image_dir>
./detr_demo detr_w16a16.adla input/
```

> **说明：** 若使用 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/detr_demo` 及对应的 32 位库路径。请将 `detr_w16a16.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同的方式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 方式一：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 目录必须加入系统的 `PATH`。
* 位于 `dependency/opencv/` 文件夹中的预编译 OpenCV。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

将下载的工具链加入 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本能够找到编译器和神经网络依赖库。

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
cd examples/DETR/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果编译器的前缀名称不同（例如使用 `aarch64-linux-gnu` 而不是 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/detr_demo`
* 32 位：`build/linux/32/detr_demo`

#### 方式二：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake 工具链文件。
* 位于 `../../../dependency/opencv/`（相对于脚本目录）的预编译 OpenCV。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/DETR/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **说明：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/detr_demo`
* 32 位：`build/yocto/32/detr_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/detr_demo /data/local/tmp/
adb push ../model/detr_w16a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x detr_demo

# Usage: ./detr_demo <model_path> <image_dir>
./detr_demo detr_w16a16.adla input/
```

> **说明：** 请将 `detr_w16a16.adla` 替换为实际的模型文件名。若使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延和 DRAM 带宽占用。

### 检测输出

对于每张输入图像，DETR 示例会执行目标检测，并生成包含检测框的可视化结果。

结果图像保存在模型结果目录中，例如：

```text
<model_name>_result/
└── test_image_result.jpg
```

可以从设备上拉取结果图像进行检查：

```bash
adb pull <model_name>_result/test_image_result.jpg
```

![DETR 检测结果](result.png)

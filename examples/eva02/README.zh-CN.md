[EN](README.md) | **CN**

# EVA02 示例

本示例使用 AMLNN 运行 EVA02。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看分类结果。

## 目录结构

```bash
examples/eva02/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and ImageNet labels
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。完整的许可证条款请参见仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例使用的 EVA-02 模型及相关源代码由第三方提供，来源于 [EVA-02 代码仓库](https://github.com/baaivision/EVA)，按 MIT License 分发。再分发 EVA-02 模型、源代码或由其衍生的实质性部分时，请保留以下版权和许可证声明：

> MIT License
>
> Copyright (c) 2022 BAAI-Vision
>
> 特此免费授予任何人获取本软件及相关文档文件（以下称“软件”）副本的权利，并允许在不受限制的情况下处理本软件，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售本软件副本的权利，以及允许获得本软件的人这样做，但须满足以下条件：
>
> 上述版权声明和本许可声明应包含在本软件的所有副本或主要部分中。
>
> 本软件按“原样”提供，不附带任何明示或暗示的保证，包括但不限于对适销性、特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任承担责任，无论是在合同诉讼、侵权行为还是其他方面，由本软件或本软件的使用或其他交易引起或与之相关。

在再分发或商业使用之前，请查阅并遵守与具体预训练权重或数据集相关的任何附加使用条款。

本示例中的部分模型预处理和后处理代码是在 AI 辅助下开发的。如发现任何疑似的许可证或署名问题，请报告给我们，以便核查并更正。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载已准备好的 EVA02 ONNX 模型：

### [点此下载 EVA02 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/EVA02/eva02_base_sim.onnx)

下载 ImageNet 类别名称：

### [点此下载 ImageNet 标签！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/mobilenet/labels.txt)

将下载的 ONNX 模型放到 `examples/eva02/model/` 下，将标签文件放到 `examples/eva02/input/` 下。

预期路径：

```text
examples/eva02/model/eva02_base_sim.onnx
examples/eva02/input/labels.txt
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/eva02/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/eva02/py
python export_adla.py \
  --onnx ../model/eva02_base_sim.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | EVA02 `.onnx` 模型的路径。                                                                      |
| `--target-platform` | 目标平台 ID。完整的受支持平台列表见[**此处**](../../docs/mapping.md)。                          |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                      |


转换完成后，保留 AMLNN 生成的文件名。

预期模型路径：

```text
examples/eva02/model/eva02_base_sim_w16a16.adla
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
python eva02_inference.py \
    --adla ../model/eva02_base_sim_w16a16.adla \
    --image-dir ../input \
    --labels ../input/labels.txt
```

参数说明：

| 参数          | 说明                                          |
| ------------- | --------------------------------------------- |
| `--adla`      | EVA02 `.adla` 模型的路径。                    |
| `--image-dir` | 存放测试图像的目录。                          |
| `--labels`    | ImageNet 类别名称 `.txt` 文件的路径。         |

脚本会自动处理指定图像目录中的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将分类结果图保存到模型结果目录。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 用于 Android 的预编译 OpenCV 位于 `dependency/opencv/` 文件夹。

#### 1. 设置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/eva02/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建文件夹中：

* 64 位：`build/android/arm64-v8a/eva02_demo`
* 32 位：`build/android/armeabi-v7a/eva02_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

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

> **说明：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/eva02_demo` 及对应的 32 位库路径。请将 `.adla` 文件名替换为你实际生成的模型文件名。

---

### 为 Linux 构建

Linux 构建过程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 文件夹必须添加到系统的 `PATH` 中。
* 预编译的 OpenCV 位于 `dependency/opencv/` 文件夹。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

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
cd examples/eva02/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器使用不同的前缀名（例如使用 `aarch64-linux-gnu` 而不是 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建文件夹中：

* 64 位：`build/linux/64/eva02_demo`
* 32 位：`build/linux/32/eva02_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已准备 CMake Toolchain 文件。
* 预编译的 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/eva02/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **说明：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建文件夹中：

* 64 位：`build/yocto/64/eva02_demo`
* 32 位：`build/yocto/32/eva02_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

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

> **说明：** 请将 `.adla` 文件名替换为你实际生成的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理完成后提供运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 分类输出

对于每张输入图像，程序会打印 Top-5 分类结果及其对应的分数。

示例：

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

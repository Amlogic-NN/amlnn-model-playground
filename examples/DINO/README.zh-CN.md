[EN](README.md) | **CN**

# DINO 示例

本示例使用 AMLNN 运行 DINO。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看分类结果。

## 目录结构

```bash
examples/DINO/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and ImageNet labels
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例使用的 DINO ViT-S/16 骨干网络和 ImageNet 线性分类器权重由第三方提供，源自 [Facebook Research DINO 项目](https://github.com/facebookresearch/dino)。原始 DINO 项目及已发布的模型权重均按 Apache License 2.0 分发。

本示例使用的经过转换和量化的 DINO 骨干网络与线性分类器模型由我方服务器按 Apache License 2.0 分发。再分发这些模型时，请保留适用的版权、许可证和署名声明。

> Copyright (c) Facebook, Inc. and its affiliates.
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

本示例中的模型预处理和后处理代码部分由 AI 辅助生成，随后经过审阅并针对本项目进行了调整。若有内容无意中与既有工作相似并引起顾虑，请联系我们，我们会及时删除或调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载准备好的 DINO 骨干网络和线性分类器 ONNX 模型：

### [点此下载 DINO 骨干网络 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dino/dino_backbone.onnx)

### [点此下载 DINO 线性分类 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dino/dino_linear.onnx)

下载 ImageNet 类别名称：

### [点此下载 ImageNet 标签！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/mobilenet/labels.txt)

将下载的模型放到 `examples/DINO/model/` 目录下，将标签文件放到 `examples/DINO/input/` 目录下。

预期路径：

```text
examples/DINO/model/dino_backbone.onnx
examples/DINO/model/dino_linear.onnx
examples/DINO/input/labels.txt
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/DINO/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/DINO/py
python export_adla.py \
  --backbone-onnx ../model/dino_backbone.onnx \
  --classifier-onnx ../model/dino_linear.onnx \
  --dataset-path ../../../resource/classification_subset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--backbone-onnx`   | DINO 骨干网络 `.onnx` 模型的路径。                                                                                                                                 |
| `--classifier-onnx` | DINO 线性分类器 `.onnx` 模型的路径。                                                                                                                               |
| `--dataset-path`    | （可选）包含用于骨干网络量化校准的输入路径的 `.txt` 文件路径。当前 `w8a8` 骨干网络配置需要此参数。[1]                                                             |
| `--target-platform` | 目标平台 ID。支持平台的完整列表见[**此处**](../../docs/mapping.md)。                                                                                               |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                                                                                         |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"` 时（例如 `w8a16 (f16)` 或 `w16a16 (f16)`）**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置。
>
> 若未设置 `activation_dtype="f16"`，`w8a16` 和 `w16a16` 将使用 `i16` 激活，此时需要校准数据集。如果未提供数据集，AMLNN 会生成随机校准数据而不是报错，这可能会严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。
>
> 在当前 DINO 导出配置中，骨干网络使用 `w8a8`，需要校准数据集；而线性分类器使用 `w8a16 (f16)`，不需要校准数据集。

转换完成后，AMLNN 生成的文件名保持不变。

预期的模型路径：

```text
examples/DINO/model/dino_backbone_w8a8.adla
examples/DINO/model/dino_linear_w8a16.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需依赖包：`amlnn`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

```bash
python dino_inference.py \
    --backbone ../model/dino_backbone_w8a8.adla \
    --classifier ../model/dino_linear_w8a16.adla \
    --image-dir ../input \
    --labels ../input/labels.txt \
    --topk 5
```

参数说明：

| 参数           | 说明                                                                    |
| -------------------- | ---------------------------------------------------------------------- |
| `--backbone`   | 编译后的 DINO 骨干网络 `.adla` 模型的路径。                             |
| `--classifier` | 编译后的 DINO 线性分类器 `.adla` 模型的路径。                           |
| `--image-dir`        | 测试图像所在目录。                                                      |
| `--labels`           | ImageNet 类别名称 `.txt` 文件的路径。                                   |
| `--topk`             | （可选）要打印的分类结果数量。默认为 `5`。                              |

脚本会自动处理指定图像目录下的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`）。

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
cd examples/DINO/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/dino_demo`
* 32 位：`build/android/armeabi-v7a/dino_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/dino_demo /data/local/tmp/
adb push ../model/dino_backbone_w8a8.adla /data/local/tmp/
adb push ../model/dino_linear_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dino_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./dino_demo <backbone.adla> <classifier.adla> <image_dir> <labels.txt> [topk]
./dino_demo dino_backbone_w8a8.adla dino_linear_w8a16.adla input/ input/labels.txt 5
```

> **说明：** 若使用 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/dino_demo` 及对应的 32 位库路径。请将 `.adla` 文件名替换为实际生成的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 文件夹必须添加到系统的 `PATH` 中。
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
cd examples/DINO/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果编译器的前缀名称不同（例如使用 `aarch64-linux-gnu` 而不是 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/dino_demo`
* 32 位：`build/linux/32/dino_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 有可用的 CMake 工具链文件。
* 位于 `../../../dependency/opencv/`（相对于脚本目录）的预编译 OpenCV。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/DINO/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **说明：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/dino_demo`
* 32 位：`build/yocto/32/dino_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/dino_demo /data/local/tmp/
adb push ../model/dino_backbone_w8a8.adla /data/local/tmp/
adb push ../model/dino_linear_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dino_demo

# Usage: ./dino_demo <backbone.adla> <classifier.adla> <image_dir> <labels.txt> [topk]
./dino_demo dino_backbone_w8a8.adla dino_linear_w8a16.adla input/ input/labels.txt 5
```

> **说明：** 请将 `.adla` 文件名替换为实际生成的模型文件名。若使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理完成后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 分类输出

对于每张输入图像，示例会先运行 DINO 骨干网络，再运行线性分类器，并打印 Top-K 分类结果及其对应的概率。

示例：

```text
============================================================
Processing image: "fish_224x224.jpeg"
============================================================
Backbone inference time: 18.1208 ms
Classifier inference time: 0.589565 ms

    Top-5 Results:
      1. tench                 prob=0.825604
      2. neck brace            prob=0.048056
      3. shower cap            prob=0.046215
      4. eel                   prob=0.006896
      5. planetarium           prob=0.005853

============================================================
```

[EN](README.md) | **CN**

# ResNet 示例

本示例使用 AMLNN 运行 ResNet-50 图像分类。完整流程如下：

1. 准备或导出 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看分类结果。

## 目录结构

```bash
examples/resnet/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and classification labels
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码采用 Apache License 2.0 许可。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本脚本直接从 [PyTorch/Torchvision](https://pytorch.org/vision/stable/models.html) 下载 ResNet-50 模型架构与预训练权重，其采用 **BSD 3-Clause License** 许可。随附的 `labels.txt` 文件包含源自 WordNet 的 1,000 个标准 ImageNet 类别名称，此处提供仅为方便使用。

运行此脚本即表示你正在下载在 ImageNet 数据集上预训练的权重。我们不托管也不分发这些权重。用户在将这些模型用于再分发或商业用途之前，有责任自行查阅并遵守 PyTorch、Torchvision 和 ImageNet 的许可证与使用条款。

本示例中的模型预处理和后处理代码部分由 AI 生成。若有任何部分与现有工作相似并引起疑虑，请联系我们，我们将予以删除或调整。

## 1. 准备 ONNX 模型

### 导出 ONNX

安装所需的软件包：

```bash
pip install torch==2.4.1 torchvision==0.19.1 onnx
```

在 `examples/resnet/py` 下运行 ONNX 导出脚本：

```bash
cd examples/resnet/py
python onnx_download.py
```

将导出的模型放在以下路径：

```text
examples/resnet/model/resnet50.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/resnet/py` 下运行 ADLA 导出脚本：

```bash
cd examples/resnet/py
python export_adla.py \
  --onnx ../model/resnet50.onnx \
  --dataset-path ../../../resource/classification_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--onnx`            | 输入的 `.onnx` 模型路径。                                                                                                                |
| `--dataset-path`    | （可选）包含量化校准所用输入路径的 `.txt` 文件路径。使用 `w8a8` 和 `w8a16 (i16)` 时必填。[1]                                             |
| `--target-platform` | 目标平台 ID。完整的支持平台列表见[**此处**](../../docs/mapping.md)。                                                                     |
| `--output-dir`      | （可选）生成的 `.adla` 模型保存目录。默认为 `../model`。                                                                                 |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"`（例如 `w8a16 (f16)` 或 `w16a16 (f16)`）时**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置。
>
> 如果未提供数据集，AMLNN 会生成随机校准数据而不是直接报错，这可能会严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。当前 ResNet-50 模型使用 `w8a8`，因此需要校准数据集才能正确量化。

转换完成后，AMLNN 生成的模型文件名保持不变。对于采用当前 `w8a8` 配置的 `resnet50.onnx` 模型，预期的模型路径为：

```text
examples/resnet/model/resnet50_w8a8.adla
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

```bash
python resnet_inference.py \
    --adla ../model/resnet50_w8a8.adla \
    --labels ../input/labels.txt \
    --image-dir ../input
```

参数说明：

| 参数          | 说明                                                    |
| ------------- | ------------------------------------------------------- |
| `--adla`      | 编译后的 ResNet-50 模型路径，`.adla` 格式。             |
| `--labels`    | ImageNet 分类标签 `.txt` 文件路径。                     |
| `--image-dir` | 包含测试图像的目录。                                    |

该脚本会自动处理指定图像目录中的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将 Top-5 分类结果打印到控制台。

## 4. 运行 C++ 示例

### 构建 Android 版本

#### 前置条件

* 已在系统中安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* Android 版预编译 OpenCV 位于 `dependency/opencv/` 文件夹中。

#### 1. 设置环境

导出 NDK（工具链）与 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/resnet/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在与所选 Android ABI 对应的构建文件夹中：

* 64 位：`build/android/arm64-v8a/resnet_demo`
* 32 位：`build/android/armeabi-v7a/resnet_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/resnet_demo /data/local/tmp/
adb push ../model/resnet50_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x resnet_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./resnet_demo <model_path> <image_dir> <labels.txt>
./resnet_demo resnet50_w8a8.adla input/ input/labels.txt
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/resnet_demo` 及对应的 32 位库路径。请将 `resnet50_w8a8.adla` 替换为实际的模型文件名。

---

### 构建 Linux 版本

Linux 构建流程支持两种不同的方式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 方式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 文件夹必须添加到系统的 `PATH` 中。
* 预编译 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。
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
cd examples/resnet/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名不同（例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在与所选架构对应的构建文件夹中：

* 64 位：`build/linux/64/resnet_demo`
* 32 位：`build/linux/32/resnet_demo`

#### 方式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake 工具链文件。
* 预编译 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/resnet/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传递 `-s` 和 `-t` 参数。

可执行文件将生成在与所选架构对应的构建文件夹中：

* 64 位：`build/yocto/64/resnet_demo`
* 32 位：`build/yocto/32/resnet_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/resnet_demo /data/local/tmp/
adb push ../model/resnet50_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x resnet_demo

# Usage: ./resnet_demo <model_path> <image_dir> [labels.txt]
./resnet_demo resnet50_w8a8.adla input/ input/labels.txt
```

> **注意：** 请将 `resnet50_w8a8.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理结束后提供运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统与 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延与 DRAM 带宽占用。

### 分类输出

对于每张输入图像，程序会将 Top-5 分类结果及其对应分数打印到控制台。

示例输出：

```text
============================================================
Processing image 1/1: dog.jpg
============================================================
 Top-5 Results:
      1. Blenheim spaniel       score=13.029234
      2. Shih-Tzu               score=9.495882
      3. flat-coated retriever  score=9.495882
      4. Leonberg               score=8.391710
      5. Pembroke               score=8.170876
============================================================
```

[EN](README.md) | **CN**

# MobileNet 示例

本示例使用 AMLNN 运行 MobileNet。完整流程如下：

1. 准备或下载 ONNX/TFLite 模型。
2. 
3. 将 ONNX/TFLite 模型转换为 ADLA 模型。
4. 使用 ADLA 模型运行 Python 示例。
5. 使用 ADLA 模型运行 C++（Linux/Android）示例。
6. 查看分类结果。

## 目录结构

```bash
examples/mobilenet/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX, TFLite, and ADLA models here
└── py/                # Python conversion and demo scripts
```

## 许可证

本示例代码采用 Apache License 2.0 许可。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

MobileNetV2 架构与预训练权重最初来自 [PyTorch/Torchvision](https://pytorch.org/vision/stable/models.html)，其采用宽松的 **BSD 3-Clause License** 许可。

允许再分发此编译后的 `.onnx` 二进制文件，且完全符合原始 BSD 3-Clause 许可条款。下载本模型用于进一步商业使用或再分发的用户应保留此署名。

随附的 `labels.txt` 文件包含源自 WordNet 的 1,000 个标准 ImageNet 类别名称，此处提供仅为方便使用。

本示例中的模型预处理、后处理和转换代码部分由 AI 生成。若有任何部分与现有工作相似并引起疑虑，请联系我们，我们将予以处理。

## 1. 准备模型

### 方式 A：导出 ONNX

安装软件包：

```bash
pip install torch==2.4.1 torchvision==0.19.1 onnx
```

运行导出脚本：

```bash
cd examples/mobilenet/py
python onnx_download.py
```

将导出的 `mobilenet_v2.onnx` 放到 `examples/mobilenet/model/` 下。

预期路径：

```text
examples/mobilenet/model/mobilenet_v2.onnx
```

### 方式 B：下载 TFLite

下载已准备好的量化 TFLite 模型，并将其放到 `examples/mobilenet/model/` 下：

### [在此下载 MobileNetV2 TFLite 文件！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/mobilenet/mobilenet_v2_1.0_224_quant.tflite)

预期路径：

```text
examples/mobilenet/model/mobilenet_v2_1.0_224_quant.tflite
```

### 下载标签文件

为了识别分类结果，请下载与该模型关联的标签文件：

### [在此下载 `labels.txt`！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/mobilenet/labels.txt)

将下载的标签文件放到：

```text
examples/mobilenet/labels.txt
```

## 2. 将 ONNX/TFLite 转换为 ADLA

在 `examples/mobilenet/py` 下运行 ADLA 导出脚本：

```bash
cd examples/mobilenet/py
python export_adla.py \
  --model ../model/mobilenet_v2.onnx \
  --dataset-path ../../../resource/classification_subset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`           | 输入的 `.onnx` 或 `.tflite` 模型路径。                                                                                                   |
| `--dataset-path`    | （可选）包含量化校准所用输入路径的 `.txt` 文件路径。使用 `w8a8` 和 `w8a16 (i16)` 时必填。[1]                                             |
| `--target-platform` | 目标平台 ID。完整的支持平台列表见[**此处**](../../docs/mapping.md)。                                                                     |
| `--output-dir` | （可选）生成的 `.adla` 模型保存目录。默认为 `../model`。 |                                   |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"`（例如 `w8a16 (f16)` 或 `w16a16 (f16)`）时**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置。
>
> 如果未提供数据集，AMLNN 会生成随机校准数据而不是报错，这可能会严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。

转换完成后，预期的模型路径为：

```text
examples/mobilenet/model/mobilenet_v2_w8a8.adla
```

## 3. 支持的输入格式

Python 和 C++ 的 MobileNet 推理示例均支持标准图像文件、原始图像文本文件，以及直接的模型输入张量文件。

| 格式                            | 说明                                                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `.jpg`、`.jpeg`、`.png`、`.bmp` | 经过常规 MobileNet 预处理流程处理的标准图像文件。                                                                               |
| `.txt`                          | 以文本形式存储的原始 BGR `uint8` 图像像素值。*该文件不得包含形状头部或元数据。*                                                |
| `.bin`                          | 以原始二进制形式存储的最终预处理模型输入张量。直接送入推理，无需额外预处理。                                                    |
| `.qtxt`                         | 以文本形式存储的最终预处理模型输入张量。直接送入推理，无需额外预处理。                                                          |

### 标准图像输入

标准图像文件使用常规的 MobileNet 预处理流程：

```text
.jpg/.jpeg/.png/.bmp
    -> image loading
    -> resize to model input resolution
    -> BGR to RGB
    -> normalize
    -> quantize if required
    -> inference
```

### 原始 `.txt` 图像输入

`.txt` 文件表示原始 BGR `uint8` 图像像素值，取值范围为 `0` 到 `255`。

该文件**仅包含像素值**，不包含形状头部或其他元数据。图像形状假定与模型输入尺寸一致，格式为 `H x W x 3`。

处理之前，示例会将 `.txt` 文件中的数值总数与预期的输入图像元素数量进行比对：

```text
H x W x 3
```

> **注意：** `.txt` 文件必须已包含与模型输入分辨率相符的像素数量，不应包含形状头部。如果数值数量与预期的 `H x W x 3` 尺寸不匹配，该输入将被跳过。

### 直接使用 `.bin` 和 `.qtxt` 模型输入

`.bin` 和 `.qtxt` 格式包含经过预处理和量化后的最终模型输入张量。

它们不包含维度、形状头部或其他元数据。

* `.bin` 以原始二进制形式存储张量。
* `.qtxt` 以文本形式存储相同的张量数值。

张量使用 AMLNN 报告的模型输入形状和数据类型读取，并直接送入推理。

对于这些直接张量格式，不会执行图像加载、缩放、RGB 转换、归一化或量化。

支持的张量类型包括 `float32`、`int8`、`uint8` 和 `int16`（如适用）。

### 生成测试输入

可以使用独立的输入生成工具 `generate_inputs.py`，从标准图像文件生成 `.txt`、`.bin` 和 `.qtxt` 测试输入。

该工具接受 `.jpg`、`.jpeg`、`.png` 和 `.bmp` 图像作为源输入，不会读取 `.txt`、`.bin` 或 `.qtxt` 文件作为源输入。

加载 ADLA 模型仅用于获取输入张量维度、缩放系数、零点和张量类型。该工具不执行推理。

对于每张源图像，该工具会生成三个文件：

* `.txt`：以文本形式包含原始 BGR 图像像素，无头部或形状元数据。只有当源图像尺寸已与模型输入尺寸一致时，推理示例才能使用该 `.txt` 文件。
* `.bin`：以二进制格式包含最终预处理并量化后的模型输入张量。
* `.qtxt`：包含与 `.bin` 相同的最终模型输入张量，但以文本形式存储张量数值。

生成流程如下：

```text
input image
    |
    +-> original raw BGR pixels
    |       -> .txt
    |
    +-> MobileNet preprocessing
            -> quantization if required
                    |
                    +-> .bin
                    |
                    +-> .qtxt
```

在 `examples/mobilenet/py` 下运行该工具：

```bash
cd examples/mobilenet/py

python generate_inputs.py \
    --adla ../model/mobilenet_v2_w8a8.adla \
    --image-dir ../input \
    --output-dir generated_inputs
```

参数说明：

| 参数           | 说明                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--adla`       | 编译后的 `.adla` 格式 MobileNet 模型路径。加载模型仅用于获取其输入张量信息。                                               |
| `--image-dir`  | 包含待转换的源 `.jpg`、`.jpeg`、`.png` 或 `.bmp` 图像的目录。                                                              |
| `--output-dir` | （可选）生成的 `.txt`、`.bin` 和 `.qtxt` 文件保存目录。默认为 `generated_inputs`。                                         |

对于每张处理过的图像，该工具会生成：

```text
generated_inputs/
├── test_image.txt
├── test_image.bin
└── test_image.qtxt
```

> **注意：** 由于 `.txt` 不含形状元数据，只有当原始源图像尺寸已与 MobileNet 模型输入尺寸一致时，生成的 `.txt` 文件才能被推理示例使用。`.bin` 和 `.qtxt` 输出不受原始图像尺寸影响，因为生成器会将其预处理到模型输入尺寸。

## 4. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需软件包：`amlnn`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

```bash
python mobilenet_inference.py \
    --adla ../model/mobilenet_v2_w8a8.adla \
    --labels ../labels.txt \
    --image-dir ../input
```

参数说明：

| 参数          | 说明                                             |
| ------------- | ------------------------------------------------ |
| `--adla`      | 编译后的 `.adla` 格式 MobileNet 模型路径。       |
| `--labels`    | `labels.txt` 的路径。                            |
| `--image-dir` | 包含测试图像的目录。                             |

该脚本会自动处理指定图像目录中所有支持的输入文件（`.jpg`、`.jpeg`、`.png`、`.bmp`、`.bin`、`.txt`），并将结果保存到 `{model_name}_result` 文件夹。

> **注意：** 不要将 `labels.txt` 放在输入目录中，因为输入目录中的 `.txt` 文件会被视为模型输入文件。

## 5. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 已安装 **Android NDK**（建议 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 预编译的 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。

#### 1. 设置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/mobilenet/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/mobilenet_demo`
* 32 位：`build/android/armeabi-v7a/mobilenet_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/mobilenet_demo /data/local/tmp/
adb push ../model/mobilenet_v2_w8a8.adla /data/local/tmp/
adb push ../input /data/local/tmp/
adb push ../labels.txt /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x mobilenet_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./mobilenet_demo <model_path> <image_dir> <labels.txt>
./mobilenet_demo mobilenet_v2_w8a8.adla input/ labels.txt
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/mobilenet_demo` 及对应的 32 位库路径。请将 `mobilenet_v2_w8a8.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 方式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（建议 GCC 10.3）。
* 必须将工具链的 `bin/` 目录添加到系统的 `PATH` 中。
* 预编译的 OpenCV 位于 `dependency/opencv/` 文件夹中。
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
cd examples/mobilenet/cpp

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

* 64 位：`build/linux/64/mobilenet_demo`
* 32 位：`build/linux/32/mobilenet_demo`

#### 方式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 可使用 CMake Toolchain 文件。
* 预编译的 OpenCV 位于 `dependency/opencv/` 文件夹中。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/mobilenet/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量来代替传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/mobilenet_demo`
* 32 位：`build/yocto/32/mobilenet_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/mobilenet_demo /data/local/tmp/
adb push ../model/mobilenet_v2_w8a8.adla /data/local/tmp/
adb push ../input /data/local/tmp/
adb push ../labels.txt /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x mobilenet_demo

# Usage: ./mobilenet_demo <model_path> <image_dir> <labels.txt>
./mobilenet_demo mobilenet_v2_w8a8.adla input/ labels.txt
```

> **注意：** 请将 `mobilenet_v2_w8a8.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请调整可执行文件路径。

## 6. 结果

### 性能反馈

通过将日志级别设置为 `INFO`，程序会在推理后提供运行时的性能信息。控制台输出可能包括：

* **硬件信息：** 系统与 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 分类输出

对于每张输入图像，程序会打印 Top-5 分类结果及其对应分数。

示例：

```text
============================================================
Processing image 1/3: dog_224x224.jpg
============================================================
    Top-5 Results:
      1. Shih-Tzu              score=15.328343
      2. Pekinese              score=12.163782
      3. Lhasa                 score=11.570427
      4. Brabancon griffon     score=7.318048
      5. Dandie Dinmont        score=7.318048

============================================================
```

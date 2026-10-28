[EN](README.md) | **CN**

# PaddleOCR Detection v4 示例

本示例使用 AMLNN 运行 PPOCR_Det。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看文本检测结果。

## 目录结构

```bash
examples/ppocr_det_v4/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example result
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。详见仓库根目录的 [LICENSE](../../LICENSE) 文件。

本项目分发 PP-OCR 检测 v4 模型，该模型最初由百度（PaddlePaddle）开发，取自 PaddleOCR 官方仓库。模型权重和网络结构按宽松的 Apache 2.0 许可证提供。若需再分发这些模型文件或将其用于商业用途，请务必保留 Apache 2.0 许可证声明。

本示例中的模型预处理和后处理代码部分由 AI 生成。若其中有内容与既有工作相似并引起顾虑，请联系我们，我们会删除或调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载准备好的 ONNX 模型：

### [点此下载 PaddleOCR Detection v4 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/ppocr_det_v4/ppocr_det_static.onnx)

将下载的模型放到以下路径：

```text
examples/ppocr_det_v4/model/ppocr_det_static.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/ppocr_det_v4/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/ppocr_det_v4/py
python export_adla.py \
  --onnx ../model/ppocr_det_static.onnx \
  --dataset-path ../../../resource/sign_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--onnx`            | PaddleOCR Detection v4 `.onnx` 模型的路径。                                                                                              |
| `--dataset-path`    | （可选）包含量化校准所用输入路径的 `.txt` 文件路径。`w8a8` 和 `w8a16 (i16)` 需要该参数。[1]                                             |
| `--target-platform` | 目标平台 ID。完整的目标平台支持列表见[**此处**](../../docs/mapping.md)。                                                                 |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                                                              |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"`（例如 `w8a16 (f16)` 或 `w16a16 (f16)`）时**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置。
>
> 若未设置 `activation_dtype="f16"`，`w8a16` 和 `w16a16` 将使用 `i16` 激活，此时需要校准数据集。如果未提供数据集，AMLNN 不会报错，而是生成随机校准数据，这可能会严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。

预期的模型路径：

```text
examples/ppocr_det_v4/model/ppocr_det_static_w8a8.adla
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
python ppocr_det_inference.py \
    --adla ../model/ppocr_det_static_w8a8.adla \
    --image-dir ../input
```

参数说明：

| 参数          | 说明                                              |
| ------------- | ------------------------------------------------- |
| `--adla`      | PaddleOCR Detection v4 `.adla` 模型的路径。      |
| `--image-dir` | 测试图像所在目录。                                |

脚本会自动处理指定图像目录下的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将文本检测可视化结果保存到模型结果目录。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 位于 `dependency/opencv/` 文件夹中的预编译 OpenCV。

#### 1. 设置环境

导出 NDK 和 AMLNN 的路径，以便构建脚本能够找到所需的工具链和神经网络依赖库。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

```bash
cd examples/ppocr_det_v4/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/ppocrv4_det_demo`
* 32 位：`build/android/armeabi-v7a/ppocrv4_det_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/ppocrv4_det_demo /data/local/tmp/
adb push ../model/ppocr_det_static_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x ppocrv4_det_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./ppocrv4_det_demo <model_path> <image_dir>
./ppocrv4_det_demo ppocr_det_static_w8a8.adla input/
```

> **说明：** 若使用 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/ppocrv4_det_demo` 及对应的 32 位库路径。请将 `ppocr_det_static_w8a8.adla` 替换为实际生成的模型文件名。

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
cd examples/ppocr_det_v4/cpp

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

* 64 位：`build/linux/64/ppocrv4_det_demo`
* 32 位：`build/linux/32/ppocrv4_det_demo`

#### 方式二：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake 工具链文件。
* 位于 `dependency/opencv/` 文件夹中的预编译 OpenCV。

##### 2. 设置环境

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/ppocr_det_v4/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **说明：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/ppocrv4_det_demo`
* 32 位：`build/yocto/32/ppocrv4_det_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/ppocrv4_det_demo /data/local/tmp/
adb push ../model/ppocr_det_static_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x ppocrv4_det_demo

# Usage: ./ppocrv4_det_demo <model_path> <image_dir>
./ppocrv4_det_demo ppocr_det_static_w8a8.adla input/
```

> **说明：** 请将 `ppocr_det_static_w8a8.adla` 替换为实际生成的模型文件名。若使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延和 DRAM 带宽占用。

### 检测输出

对于每张输入图像，程序会检测文本区域，打印检测信息和推理耗时，并保存结果图像，其中已在图像上绘制检测到的文本区域。

结果图像保存到模型结果目录。

示例结果路径：

```text
ppocr_det_static_w8a8_result/
└── test_image_result.jpg
```

可以从设备上拉取结果图像进行查看：

```bash
adb pull ppocr_det_static_w8a8_result/test_image_result.jpg
```

![PP-OCR 检测 v4 检测结果](./result.jpg)

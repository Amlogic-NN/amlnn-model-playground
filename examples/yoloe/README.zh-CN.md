[EN](README.md) | **CN**

# YOLOE 示例

本示例使用 AMLNN 运行 YOLOE。完整流程如下：

1. 将 ONNX 模型转换为 ADLA 模型。
2. 使用 ADLA 模型运行 Python 示例。
3. 使用 ADLA 模型运行 C++（Linux/Android）示例。
4. 查看检测图像/结果。

## 目录结构

```bash
examples/yoloe/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example detection output
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。详见仓库根目录的 [LICENSE](../../LICENSE) 文件。

Yoloe-v8s 模型由第三方提供，模型权重不在本仓库中分发。在自行再分发或商用之前，请查阅并遵守模型来源方的许可证与使用条款。不过，推理所需的配套标签文件由本仓库一并分发；关于标签的具体使用条款，请参考原始来源数据集。

本示例中的模型预处理和后处理代码部分由 AI 生成。若其中某部分与既有工作相似并引起顾虑，请联系我们，我们会将其删除或调整。

## 标签下载

要正确识别检测到的目标，您需要该模型对应的专用标签文件。

### [在此下载 `labels.txt`！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/yoloe/labels.txt)

将下载的标签文件放到：

```text
examples/yoloe/input/labels.txt
```

## 1. 模型转换

在 `examples/yoloe/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/yoloe/py
python export_adla.py \
  --onnx ../model/model.onnx \
  --dataset-path ../../../resource/detection_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--onnx`            | 输入 `.onnx` 模型的路径。                                                                                                                                                         |
| `--dataset-path`    | （可选）`.txt` 文件路径，其中列出用于量化校准的输入图像路径。使用 `w8a8` 和 `w8a16 (i16)` 配置时必需，以保证量化正确。[1]                                                       |
| `--target-platform` | 目标平台 ID。完整的目标平台支持列表见[**此处**](../../docs/mapping.md)。                                                                                                          |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                                                                                                        |

> **[1] 量化数据集说明**
>
> 当使用 `activation_dtype="f16"` 时（例如 `w8a16 (f16)` 或 `w16a16 (f16)`），**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置该参数。
>
> 如果未提供数据集，AMLNN 不会报错，而是生成随机校准数据，这可能导致模型精度严重下降。
>
> `activation_dtype` 不适用于 `w8a8`。当前 YOLOE 模型使用 `w8a8`，因此需要校准数据集才能正确完成量化。

转换完成后，AMLNN 生成的文件名保持不变。对于 `model.onnx`，在当前 `w8a8` 配置下，预期的模型路径为：

```text
examples/yoloe/model/model_w8a8.adla
```

## 2. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需软件包：`amlnn`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

```bash
python yoloe_inference.py \
    --adla ../model/model_w8a8.adla \
    --image-dir ../input \
    --labels ../input/labels.txt
```

参数说明：

| 参数          | 说明                                                              |
| ------------- | ----------------------------------------------------------------- |
| `--adla`      | YOLOE `.adla` 模型的路径。                                        |
| `--image-dir` | 测试图像所在目录。                                                |
| `--labels`    | （可选）`labels.txt` 的路径，默认为 `../input/labels.txt`。       |
| `--conf`      | （可选）置信度阈值，默认为 `0.7`。                                |
| `--nms`       | （可选）非极大值抑制使用的 IoU 阈值，默认为 `0.05`。              |

脚本会自动处理指定图像目录下的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将检测结果保存到 `{model_name}_result` 文件夹。

## 3. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* **Android NDK**（推荐 r27d）已安装在系统中。
* 已下载并解压 **AMLNN Toolkit**。
* 预编译的 Android 版 OpenCV 位于 `dependency/opencv/` 文件夹。

#### 1. 设置环境

导出 NDK（工具链）与 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/yoloe/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/yoloe_demo`
* 32 位：`build/android/armeabi-v7a/yoloe_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/yoloe_demo /data/local/tmp/
adb push ../model/model_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x yoloe_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./yoloe_demo <model_path> <image_dir> <labels.txt>
./yoloe_demo model_w8a8.adla input/ input/labels.txt
```

> **说明：** 若使用 32 位（`armeabi-v7a`）构建，请改用 `build/android/armeabi-v7a/yoloe_demo` 及对应的 32 位库路径。请将 `model_w8a8.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 目录必须加入系统 `PATH`。
* 预编译的 OpenCV 位于 `dependency/opencv/` 文件夹。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

将下载的工具链加入 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本能够找到编译器和神经网络依赖。

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
cd examples/yoloe/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果编译器的前缀名不同（例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/yoloe_demo`
* 32 位：`build/linux/32/yoloe_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake Toolchain 文件。
* 预编译的 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。

##### 1. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 2. 构建

```bash
cd examples/yoloe/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **说明：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/yoloe_demo`
* 32 位：`build/yocto/32/yoloe_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/yoloe_demo /data/local/tmp/
adb push ../model/model_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x yoloe_demo

# Usage: ./yoloe_demo <model_path> <image_dir> <labels.txt>
./yoloe_demo model_w8a8.adla input/ input/labels.txt
```

> **说明：** 请将 `model_w8a8.adla` 替换为实际的模型文件名。若使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 4. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包含：

* **硬件信息：** 系统版本和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延和 DRAM 带宽占用。

### 检测输出

示例会对每张输入图像执行目标检测，并保存包含检测框的可视化结果。

结果图像以输入图像名保存在模型结果目录中。例如：

```text
<model_name>_result/
└── test_image_result.jpg
```

可以从设备上拉取结果图像进行查看：

```bash
adb pull <model_name>_result/test_image_result.jpg
```

![YOLOE 检测结果](./result.jpg)

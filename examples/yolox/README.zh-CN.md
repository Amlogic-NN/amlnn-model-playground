[EN](README.md) | **CN**

# YOLOX 示例

本示例使用 AMLNN 运行 YOLOX。完整流程如下：

1. 将 ONNX 模型转换为 ADLA 模型。
2. 使用 ADLA 模型运行 Python 示例。
3. 使用 ADLA 模型运行 C++（Linux/Android）示例。
4. 查看检测图像/结果。

## 目录结构

```bash
examples/yolox/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example detection output
```

## 许可证

本示例代码的许可证为 Apache License 2.0，详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

Yolox 模型由第三方提供，在重新分发或商业使用前，请查阅并遵守模型来源的许可证与使用条款。

本示例中的模型预处理和后处理代码部分由 AI 生成。如果任何部分与现有工作相似并引起疑虑，请联系我们，我们将及时删除或调整。

## 1. 模型转换

在 `examples/yolox/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/yolox/py
python export_adla.py \
  --onnx ../model/model.onnx \
  --dataset-path ../../../resource/detection_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--onnx`            | 输入 `.onnx` 模型的路径。                                                                                                                                                            |
| `--dataset-path`    | （可选）包含用于量化校准的输入图像路径的 `.txt` 文件路径。使用 `w8a8` 和 `w8a16 (i16)` 配置时需提供该文件，以保证量化正确。[1]                                                     |
| `--target-platform` | 目标平台 ID。支持的平台完整列表见 [**此处**](../../docs/mapping.md)。                                                                                                                |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                                                                                                           |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"`（例如 `w8a16 (f16)` 或 `w16a16 (f16)`）时**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置该参数。
>
> 如果未提供数据集，AMLNN 会生成随机校准数据而不会报错，这可能严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。当前 YOLOX 模型使用 `w8a8`，因此需要校准数据集才能正确量化。

转换完成后，AMLNN 生成的模型文件名保持不变。对于示例 `model.onnx`，在当前 `w8a8` 配置下，预期的模型路径为：

```text
examples/yolox/model/model_w8a8.adla
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
python yolox_inference.py \
    --adla ../model/model_w8a8.adla \
    --image-dir ../input
```

参数说明：

| 参数          | 说明                                                                              |
| ------------- | --------------------------------------------------------------------------------- |
| `--adla`      | 编译后的 `.adla` 格式模型路径。                                                   |
| `--image-dir` | 测试图像所在目录。                                                                |
| `--conf`      | （可选）用于过滤检测结果的置信度阈值。默认为 `0.4`。                              |
| `--nms`       | （可选）非极大值抑制期间使用的 IoU 阈值。默认为 `0.45`。                          |

脚本会自动处理指定图像目录中的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将结果保存到 `{model_name}_result` 文件夹。

## 3. 运行 C++ 示例

### Android 构建

#### 前置条件

* 已在系统中安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 位于 `dependency/opencv/` 文件夹中的 Android 预编译 OpenCV。

#### 1. 环境准备

导出 NDK（工具链）和 AMLNN（神经网络依赖库）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/yolox/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在对应所选 Android ABI 的构建目录中：

* 64 位：`build/android/arm64-v8a/yolox_demo`
* 32 位：`build/android/armeabi-v7a/yolox_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/yolox_demo /data/local/tmp/
adb push ../model/model_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x yolox_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./yolox_demo <model_path> <image_dir>
./yolox_demo model_w8a8.adla input/
```

> **注意：** 32 位（`armeabi-v7a`）构建请使用 `build/android/armeabi-v7a/yolox_demo` 及对应的 32 位库路径。请将 `model_w8a8.adla` 替换为实际的模型文件名。

---

### Linux 构建

Linux 构建流程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 文件夹必须添加到系统的 `PATH` 中。
* 位于 `dependency/opencv/` 文件夹中的预编译 OpenCV。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 环境准备

将下载好的工具链添加到 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本找到编译器和神经网络依赖库。

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
cd examples/yolox/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果编译器前缀名称不同（例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在对应所选架构的构建目录中：

* 64 位：`build/linux/64/yolox_demo`
* 32 位：`build/linux/32/yolox_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 拥有可用的 CMake Toolchain 文件。
* 位于 `../../../dependency/opencv/`（相对于脚本目录）的预编译 OpenCV。

##### 2. 环境准备

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/yolox/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 你也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 标志。

可执行文件将生成在对应所选架构的构建目录中：

* 64 位：`build/yocto/64/yolox_demo`
* 32 位：`build/yocto/32/yolox_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/yolox_demo /data/local/tmp/
adb push ../model/model_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x yolox_demo

# Usage: ./yolox_demo <model_path> <image_dir>
./yolox_demo model_w8a8.adla input/
```

> **注意：** 请将 `model_w8a8.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请调整可执行文件路径。

## 4. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理完成后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 检测输出

示例会对每张输入图像执行目标检测，并保存包含检测框的可视化结果。

结果图像以输入图像名称保存在模型结果目录中。例如：

```text
<model_name>_result/
└── test_image_result.jpg
```

你可以从设备上拉取结果图像进行查看：

```bash
adb pull <model_name>_result/test_image_result.jpg
```

![YOLOX 检测结果](./result.png)

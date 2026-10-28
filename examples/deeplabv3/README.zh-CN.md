[EN](README.md) | **CN**

# DeeplabV3 示例

本示例使用 AMLNN 运行 DeeplabV3。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看分割图像/结果。

## 目录结构

```bash
examples/deeplabv3/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.png         # Example segmentation output
```

## 许可证

本示例代码采用 Apache License 2.0 许可。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例提供的 DeepLabV3 模型来自第三方。其权重随 PyTorch/Torchvision 项目一同分发。为便于配合本工具链使用，我将转换后的 `.onnx` 模型托管在自己的服务器上分发。在再分发或商业使用前，请查阅并遵守原始模型来源的许可证与使用条款：

> **第三方模型许可证：**
> PyTorch/Torchvision 项目（包括 DeepLabV3 架构和预训练权重）采用 **BSD 3-Clause License** 分发。完整的许可证条款可在 Torchvision 官方仓库查看：https://github.com/pytorch/vision/blob/main/LICENSE

本示例中的模型预处理和后处理代码部分由 AI 生成。如果其中有任何部分与现有工作相似并引起疑虑，请联系我们，我们会及时删除或调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载准备好的 ONNX 模型，并将其放到 `examples/deeplabv3/model/` 下：

- [在此下载 DeepLabV3 ONNX 文件！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/deeplabv3/deeplabv3_mobilenet_v3_sim.onnx)

预期路径：

```text
examples/deeplabv3/model/deeplabv3_mobilenet_v3_sim.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/deeplabv3/py` 下运行 ADLA 导出脚本：

```bash
cd examples/deeplabv3/py
python export_adla.py \
  --onnx ../model/deeplabv3_mobilenet_v3_sim.onnx \
  --dataset-path ../../../resource/detection_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `--onnx`            | 输入 `.onnx` 模型的路径。                                                                                                               |
| `--dataset-path`    | （可选）包含量化校准所用输入路径的 `.txt` 文件路径。`w8a8` 和 `w8a16 (i16)` 需要此项。[1]                                               |
| `--target-platform` | 目标平台 ID。支持的平台完整列表见[**此处**](../../docs/mapping.md)。                                                                     |
| `--output-dir`      | （可选）生成的 `.adla` 模型保存目录。默认为 `../model`。                                                                                |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"` 时（如 `w8a16 (f16)` 或 `w16a16 (f16)`），**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置该参数。
>
> 如果未提供数据集，AMLNN 不会报错，而是生成随机校准数据，这可能会严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。

转换完成后，预期的模型路径为：

```text
examples/deeplabv3/model/deeplabv3_mobilenet_v3_sim_w8a8.adla
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
python dlv3_inference.py \
    --adla ../model/deeplabv3_mobilenet_v3_sim_w8a8.adla \
    --image-dir ../input
```

参数说明：

| 参数          | 说明                                                   |
| ------------- | ------------------------------------------------------ |
| `--adla`      | 编译后的 DeepLabV3 模型路径，格式为 `.adla`。          |
| `--image-dir` | 测试图像所在目录。                                     |

脚本会自动处理指定图像目录中的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将结果保存到 `{model_name}_result` 文件夹中。

## 4. 运行 C++ 示例

### 构建 Android 版本

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 位于 `dependency/opencv/` 文件夹中的 Android 预编译 OpenCV。

#### 1. 设置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/deeplabv3/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成到与所选 Android ABI 对应的构建文件夹中：

* 64 位：`build/android/arm64-v8a/dlv3_demo`
* 32 位：`build/android/armeabi-v7a/dlv3_demo`

#### 3. 示例运行

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/dlv3_demo /data/local/tmp/
adb push ../model/deeplabv3_mobilenet_v3_sim_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dlv3_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./dlv3_demo <model_path> <image_dir>
./dlv3_demo deeplabv3_mobilenet_v3_sim_w8a8.adla input/
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/dlv3_demo` 及对应的 32 位库路径。请将 `deeplabv3_mobilenet_v3_sim_w8a8.adla` 替换为实际的模型文件名。

---

### 构建 Linux 版本

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 方式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 必须将工具链的 `bin/` 文件夹添加到系统的 `PATH` 中。
* 位于 `dependency/opencv/` 文件夹中的预编译 OpenCV。
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
cd examples/deeplabv3/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果编译器的前缀名称不同（例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成到与所选架构对应的构建文件夹中：

* 64 位：`build/linux/64/dlv3_demo`
* 32 位：`build/linux/32/dlv3_demo`

#### 方式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake 工具链文件。
* 位于 `../../../dependency/opencv/` 的预编译 OpenCV（相对于脚本目录）。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/deeplabv3/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成到与所选架构对应的构建文件夹中：

* 64 位：`build/yocto/64/dlv3_demo`
* 32 位：`build/yocto/32/dlv3_demo`

#### 3. 示例运行

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/dlv3_demo /data/local/tmp/
adb push ../model/deeplabv3_mobilenet_v3_sim_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dlv3_demo

# Usage: ./dlv3_demo <model_path> <image_dir>
./dlv3_demo deeplabv3_mobilenet_v3_sim_w8a8.adla input/
```

> **注意：** 请将 `deeplabv3_mobilenet_v3_sim_w8a8.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理完成后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 分割输出

对于每张输入图像，DeepLabV3 示例会执行语义分割，并生成分割结果的可视化图像。

结果图像保存在模型结果目录中。

可以从设备拉取结果图像进行检查：

```bash
adb pull <model_name>_result/test_image_result.png
```

![DeepLabV3 分割结果](./test_image.png)

[EN](README.md) | **CN**

# PaddlePaddle PP-LiteSeg 示例

本示例使用 AMLNN 运行 PaddlePaddle PP-LiteSeg。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看语义分割结果。

## 目录结构

```bash
examples/ppseg/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example segmentation output
```

## 许可证

本示例代码采用 Apache License 2.0 许可。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

我们在本仓库中分发 PP-LiteSeg `.onnx` 模型文件。该模型源自 PaddlePaddle/PaddleSeg 项目，该项目同样采用 Apache License 2.0 许可。在自行再分发或商业使用前，请查阅并遵守模型来源的许可证与使用条款。

本示例中的模型预处理和后处理代码部分由 AI 生成。如果其中有任何部分与现有工作相似并引起疑虑，请联系我们，我们会及时删除或调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载已准备好的 ONNX 模型：

### [在此下载 PP-LiteSeg ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/ppseg/pp_liteseg.onnx)

将下载的模型放到以下位置：

```text
examples/ppseg/model/pp_liteseg.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/ppseg/py` 下运行 ADLA 导出脚本：

```bash
cd examples/ppseg/py
python export_adla.py \
  --onnx ../model/pp_liteseg.onnx \
  --dataset-path ../../../resource/detection_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--onnx`            | 输入 `.onnx` 模型的路径。                                                                                                                |
| `--dataset-path`    | （可选）包含量化校准所用输入路径的 `.txt` 文件路径。`w8a8` 和 `w8a16 (i16)` 需要此项。[1]                                                |
| `--target-platform` | 目标平台 ID。支持的平台完整列表见[**此处**](../../docs/mapping.md)。                                                                      |
| `--output-dir`      | （可选）生成的 `.adla` 模型保存目录。默认为 `../model`。                                                                                 |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"` 时（如 `w8a16 (f16)` 或 `w16a16 (f16)`），**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置该参数。
>
> 如果未提供数据集，AMLNN 不会报错，而是生成随机校准数据，这可能会严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。当前 PP-LiteSeg 模型使用 `w8a8`，因此需要校准数据集才能正确量化。

转换完成后，AMLNN 生成的模型文件名会保持不变。对于采用当前 `w8a8` 配置的 `pp_liteseg.onnx` 模型，预期模型路径为：

```text
examples/ppseg/model/pp_liteseg_w8a8.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 需要安装的软件包：`amlnn`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

```bash
python ppseg_inference.py \
    --adla ../model/pp_liteseg_w8a8.adla \
    --image-dir ../input
```

参数说明：

| 参数           | 说明                                                                           |
| -------------- | ------------------------------------------------------------------------------ |
| `--adla`       | 编译好的 `.adla` 格式 PP-LiteSeg 模型的路径。                                  |
| `--image-dir`  | 包含测试图像的目录。                                                           |
| `--mask-alpha` | （可选）控制掩码的透明度（不透明度）。默认为 `0.5`。                           |

脚本会自动处理指定图像目录中的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将语义分割可视化结果保存到 `{model_name}_result` 文件夹中。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 已安装 **Android NDK**（推荐 r27d）。
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
cd examples/ppseg/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的 build 文件夹中：

* 64 位：`build/android/arm64-v8a/ppseg_demo`
* 32 位：`build/android/armeabi-v7a/ppseg_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/ppseg_demo /data/local/tmp/
adb push ../model/pp_liteseg_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x ppseg_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./ppseg_demo <model_path> <image_dir>
./ppseg_demo pp_liteseg_w8a8.adla input/
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/ppseg_demo` 及对应的 32 位库路径。请将 `pp_liteseg_w8a8.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

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
cd examples/ppseg/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名称不同（例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的 build 文件夹中：

* 64 位：`build/linux/64/ppseg_demo`
* 32 位：`build/linux/32/ppseg_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已获取 CMake 工具链文件。
* 位于 `../../../dependency/opencv/`（相对于脚本目录）的预编译 OpenCV。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/ppseg/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的 build 文件夹中：

* 64 位：`build/yocto/64/ppseg_demo`
* 32 位：`build/yocto/32/ppseg_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/ppseg_demo /data/local/tmp/
adb push ../model/pp_liteseg_w8a8.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x ppseg_demo

# Usage: ./ppseg_demo <model_path> <image_dir>
./ppseg_demo pp_liteseg_w8a8.adla input/
```

> **注意：** 请将 `pp_liteseg_w8a8.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统与 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延和 DRAM 带宽占用。

### 分割输出

对于每张输入图像，示例会执行语义分割，并保存一张将分割掩码叠加在输入图像上的可视化结果。

结果图像会以输入图像名称保存在模型结果目录中。例如：

```text
<model_name>_result/
└── test_image_result.jpg
```

你可以从设备上拉取结果图像以进行检查：

```bash
adb pull <model_name>_result/test_image_result.jpg
```

![PP-LiteSeg 分割结果](./result.jpg)

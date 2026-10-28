[EN](README.md) | **CN**

# DINOv2 深度估计示例

本示例使用 AMLNN 运行 DINOv2 深度估计。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看深度估计结果。

## 目录结构

```bash
examples/DINOv2_depth_estimation/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.png         # Example depth estimation output
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。完整的许可证条款请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例使用的 DINOv2 ViT-B/14 主干网络和 NYU Depth V2 DPT 深度估计头来自 [Meta AI DINOv2 项目](https://github.com/facebookresearch/dinov2)。原始 DINOv2 代码和模型权重以 Apache License 2.0 发布。

本示例分发的 ONNX 和 ADLA 模型文件是原始 DINOv2 主干与 DPT 深度头权重的转换版本，并在适用情况下经过编译或量化处理。这些模型文件相较于原始发布版本，经过了模型导出、计算图转换、主干与深度头拆分、编译和/或量化等修改。

转换后的模型文件按 Apache License 2.0 再分发。再分发这些文件时，您必须：

* 向接收方提供一份 Apache License 2.0 副本；
* 保留适用的版权、专利、商标和署名声明；
* 保留本声明或等效的、标明原始 DINOv2 项目的声明；以及
* 明确说明这些模型文件是从原始模型权重转换或以其他方式修改而来。

> Copyright (c) Meta Platforms, Inc. and affiliates.
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

本示例中的预处理、推理和后处理实现是在 AI 辅助下为本项目开发的，随后由项目维护者审阅并调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载已准备好的 DINOv2 主干和 NYU DPT 深度头 ONNX 模型：

### [点此下载 DINOv2 主干 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dinov2_dd/dinov2_vitb14_backbone_sim.onnx)

### [点此下载 DINOv2 NYU DPT ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/dinov2_dd/dinov2_vitb14_nyu_dpthead_sim.onnx)

将下载的 ONNX 模型放到 `examples/DINOv2_depth_estimation/model/` 目录下。

预期路径：

```text
examples/DINOv2_depth_estimation/model/dinov2_vitb14_backbone_sim.onnx
examples/DINOv2_depth_estimation/model/dinov2_vitb14_nyu_dpthead_sim.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/DINOv2_depth_estimation/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/DINOv2_depth_estimation/py
python export_adla.py \
  --backbone-onnx ../model/dinov2_vitb14_backbone_sim.onnx \
  --depth-onnx ../model/dinov2_vitb14_nyu_dpthead_sim.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--backbone-onnx`   | DINOv2 主干 `.onnx` 模型的路径。                                                                |
| `--depth-onnx`      | DINOv2 深度预测头 `.onnx` 模型的路径。                                                          |
| `--target-platform` | 目标平台 ID。完整的目标平台支持列表见[**此处**](../../docs/mapping.md)。                        |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                      |

转换完成后，AMLNN 生成的文件名保持不变。

在当前 `w16a16` 配置下，预期的模型路径为：

```text
examples/DINOv2_depth_estimation/model/dinov2_vitb14_backbone_sim_w16a16.adla
examples/DINOv2_depth_estimation/model/dinov2_vitb14_nyu_dpthead_sim_w16a16.adla
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
python dinov2_de_inference.py \
    --backbone ../model/dinov2_vitb14_backbone_sim_w16a16.adla \
    --depth ../model/dinov2_vitb14_nyu_dpthead_sim_w16a16.adla \
    --image-dir ../input \
    --output-dir depth_results \
    --min-depth 0.001 \
    --max-depth 10.0
```

参数说明：

| 参数           | 说明                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| `--backbone` | DINOv2 主干 `.adla` 模型的路径。                                    |
| `--depth`    | 单输入 DINOv2 深度头 `.adla` 模型的路径。                           |
| `--image-dir`      | 测试图像所在目录。                                                  |
| `--output-dir`     | （可选）深度结果的保存目录。默认为 `depth_results`。                |
| `--min-depth`      | （可选）NYU 深度值下限，单位为米。默认为 `0.001`。                  |
| `--max-depth`      | （可选）NYU 深度值上限，单位为米。默认为 `10.0`。                   |

脚本会自动处理指定图像目录下的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将生成的深度结果保存到指定的输出目录。

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
cd examples/DINOv2_depth_estimation/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/dinov2_de_demo`
* 32 位：`build/android/armeabi-v7a/dinov2_de_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/dinov2_de_demo /data/local/tmp/
adb push ../model/dinov2_vitb14_backbone_sim_w16a16.adla /data/local/tmp/
adb push ../model/dinov2_vitb14_nyu_dpthead_sim_w16a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dinov2_de_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./dinov2_de_demo <backbone.adla> <depth.adla> <image_dir> [min_depth] [max_depth]
./dinov2_de_demo dinov2_vitb14_backbone_sim_w16a16.adla dinov2_vitb14_nyu_dpthead_sim_w16a16.adla input/ 0.001 10.0
```

> **说明：** 若使用 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/dinov2_de_demo` 及对应的 32 位库路径。请将 `.adla` 文件名替换为实际生成的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

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
cd examples/DINOv2_depth_estimation/cpp

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

* 64 位：`build/linux/64/dinov2_de_demo`
* 32 位：`build/linux/32/dinov2_de_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

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
cd examples/DINOv2_depth_estimation/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **说明：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/dinov2_de_demo`
* 32 位：`build/yocto/32/dinov2_de_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/dinov2_de_demo /data/local/tmp/
adb push ../model/dinov2_vitb14_backbone_sim_w16a16.adla /data/local/tmp/
adb push ../model/dinov2_vitb14_nyu_dpthead_sim_w16a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x dinov2_de_demo

# Usage: ./dinov2_de_demo <backbone.adla> <depth.adla> <image_dir> [min_depth] [max_depth]
./dinov2_de_demo dinov2_vitb14_backbone_sim_w16a16.adla dinov2_vitb14_nyu_dpthead_sim_w16a16.adla input/ 0.001 10.0
```

> **说明：** 请将 `.adla` 文件名替换为实际生成的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理完成后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 深度估计输出

对于每张输入图像，示例会先运行 DINOv2 主干，再运行 NYU DPT 深度估计头，并生成深度可视化结果。

Python 示例将生成的深度结果保存到 `--output-dir` 指定的目录，该参数默认为：

```text
depth_results/
```

示例输入图像：

![DINOv2 深度输入](./input/fox.jpg)

示例深度估计结果：

![DINOv2 深度估计结果](./result.png)

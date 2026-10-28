[EN](README.md) | **CN**

# Blazepose_landmark 示例

本示例使用 AMLNN 运行 Blazepose_landmark。完整流程如下：

1. 准备或下载 TFLite 模型。
2. 将 TFLite 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看关键点图像/结果。

## 目录结构

```bash
examples/blazepose_landmark/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and detector output files
├── model/             # Put TFLite and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example landmark output
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。详见仓库根目录的 [LICENSE](../../LICENSE) 文件。

BlazePose 检测与关键点架构及预训练权重最初来自 [Google 的 MediaPipe 项目](https://github.com/google/mediapipe)，该项目基于 **Apache License 2.0** 许可发布。

托管和再分发这些 `.tflite` 模型二进制文件是允许的，且完全符合原始 Apache License 2.0 条款。用户下载这些模型用于进一步商业使用或再分发时，应保留此署名并附带原始许可证条款。

本示例中的模型预处理、后处理和转换代码部分由 AI 生成。若其中任何部分与现有工作相似并引起疑虑，请联系我们，我们将予以处理。

## 1. 准备 TFLite 模型

### 下载 TFLite

下载预先准备好的 TFLite 模型，并放到 `examples/blazepose_landmark/model/` 下：

### [在此下载 Blazepose Landmark TFLite 文件！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/blazepose_landmark/pose_landmark_full_float32.tflite)

预期路径：

```text
examples/blazepose_landmark/model/pose_landmark_full_float32.tflite
```

## 2. 将 TFLite 转换为 ADLA

在 `examples/blazepose_landmark/py` 下运行 ADLA 导出脚本：

```bash
cd examples/blazepose_landmark/py
python export_adla.py \
  --tflite ../model/pose_landmark_full_float32.tflite \
  --dataset-path ../../../resource/pose_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--tflite`          | 输入 `.tflite` 模型的路径。                                                                                                    |
| `--dataset-path`    | 包含用于量化校准的输入路径的 `.txt` 文件路径。`w8a8` 和 `w8a16 (i16)` 必需。[1]                                                |
| `--target-platform` | 目标平台 ID。支持的平台完整列表见 [**此处**](../../docs/mapping.md)。                                                          |
| `--output-dir`            | 生成的 `.adla` 模型保存目录。可选；默认为 `../model`。                                                                |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"`（例如 `w8a16 (f16)` 或 `w16a16 (f16)`）时**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置。
>
> 如果未提供数据集，AMLNN 会生成随机校准数据而不会报错，这可能会严重降低模型精度。
>
> `activation_dtype` 不适用于 `w8a8`。

转换完成后，预期的模型路径为：

```text
examples/blazepose_landmark/model/pose_landmark_full_float32_w8a8.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 依赖包：`amlnn`

### 安装依赖

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

关键点模型需要 BlazePose 检测示例生成的检测输出。每张输入图像在检测结果目录中必须有对应的 `*_det.txt` 文件，否则该图像将被跳过。

```bash
python blazepose_landmark_inference.py \
    --adla ../model/pose_landmark_full_float32_w8a8.adla \
    --image-dir ../input \
    --detections-dir ../input \
    --conf 0.5
```

参数说明：

| 参数               | 说明                                                                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `--adla`           | 编译好的 `.adla` 格式关键点模型路径。                                                                                                          |
| `--image-dir`      | 测试图像所在目录。每张图像在检测结果目录中必须有对应的 `*_det.txt` 文件，否则将被跳过。                                                       |
| `--detections-dir` | 存放 BlazePose 检测示例生成的 `*_det.txt` 文件的目录。                                                                                         |
| `--conf`           | （可选）用于过滤关键点结果的置信度阈值。默认为 `0.5`。                                                                                         |

脚本会自动处理指定图像目录中所有支持的图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将结果保存到 `{model_name}_result` 文件夹。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* `dependency/opencv/` 文件夹中已预编译的 Android 版 OpenCV。

#### 1. 配置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/blazepose_landmark/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的 build 文件夹中：

* 64 位：`build/android/arm64-v8a/blazepose_landmark_demo`
* 32 位：`build/android/armeabi-v7a/blazepose_landmark_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/blazepose_landmark_demo /data/local/tmp/
adb push ../model/pose_landmark_full_float32_w8a8.adla /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x blazepose_landmark_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./blazepose_landmark_demo <model.adla> <image_dir> <detections_dir>
./blazepose_landmark_demo pose_landmark_full_float32_w8a8.adla input/ input/
```

> **注意：** 32 位（`armeabi-v7a`）构建请使用 `build/android/armeabi-v7a/blazepose_landmark_demo` 及对应的 32 位库路径。请将 `pose_landmark_full_float32_w8a8.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 必须将工具链的 `bin/` 文件夹添加到系统的 `PATH` 中。
* `dependency/opencv/` 文件夹中已预编译的 OpenCV。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 配置环境

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
cd examples/blazepose_landmark/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名称不同（例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的 build 文件夹中：

* 64 位：`build/linux/64/blazepose_landmark_demo`
* 32 位：`build/linux/32/blazepose_landmark_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake Toolchain 文件。
* 预编译的 OpenCV 位于 `../../../dependency/opencv/`（相对于脚本目录）。

##### 2. 配置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/blazepose_landmark/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不传 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的 build 文件夹中：

* 64 位：`build/yocto/64/blazepose_landmark_demo`
* 32 位：`build/yocto/32/blazepose_landmark_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/blazepose_landmark_demo /data/local/tmp/
adb push ../model/pose_landmark_full_float32_w8a8.adla /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x blazepose_landmark_demo

# Usage: ./blazepose_landmark_demo <model.adla> <image_dir> <detections_dir>
./blazepose_landmark_demo pose_landmark_full_float32_w8a8.adla input/ input/
```

> **注意：** 请将 `pose_landmark_full_float32_w8a8.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包含：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 关键点输出

BlazePose 关键点示例使用 BlazePose 检测阶段生成的 `*_det.txt` 文件，以确定哪些检测到的姿态应交给关键点模型处理。

生成的人体姿态关键点可视化结果会保存到模型结果目录。例如：

```text
<model_name>_result/test_image_result.jpg
```

你可以从设备上拉取结果图像进行检查：

```bash
adb pull <model_name>_result/test_image_result.jpg
```

![BlazePose 关键点结果](./result.jpg)

> **注意：** BlazePose 关键点推理依赖 BlazePose 检测示例生成的检测输出。没有对应 `*_det.txt` 文件的输入图像将被跳过。

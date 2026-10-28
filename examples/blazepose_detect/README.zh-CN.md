[EN](README.md) | **CN**

# Blazepose_detect 示例

本示例使用 AMLNN 运行 Blazepose_detect。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看检测图像/结果。

## 目录结构

```bash
examples/blazepose_detect/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example detection output
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。详见仓库根目录的 [LICENSE](../../LICENSE) 文件。

BlazePose 检测与关键点架构及预训练权重最初来自 [Google 的 MediaPipe 项目](https://github.com/google/mediapipe)，该项目采用 **Apache License 2.0** 许可。

托管和再分发这些 `.tflite` 模型二进制文件是允许的，且完全符合 Apache License 2.0 的原始条款。用户下载这些模型用于后续商业用途或再分发时，应保留此署名并附上原始许可证条款。

本示例中的模型预处理、后处理及转换代码部分由 AI 生成。若其中有内容与既有工作相似而引起顾虑，请联系我们，我们会及时处理。

## 1. 准备 ONNX 模型

### 下载 TFLite

下载准备好的 TFLite 模型，放到 `examples/blazepose_detect/model/` 目录下：


- [点此下载 Blazepose Detect TFLite 文件！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/blazepose_detect/pose_detection_float32.tflite)


预期路径：

```text
examples/blazepose_detect/model/pose_detection_float32.tflite
```

## 2. 将 ONNX/TFLite 转换为 ADLA

在 `examples/blazepose_detect/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/blazepose_detect/py
python export_adla.py \
  --tflite ../model/pose_detection_float32.tflite \
  --dataset-path ../../../resource/pose_dataset.txt \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `--tflite`          | 输入 `.tflite` 模型的路径。                                                                                                 |
| `--dataset-path` | 包含量化校准所用输入路径的 `.txt` 文件路径。`w8a8` 和 `w8a16 (i16)` 必需。[1] |
| `--target-platform` | 目标平台 ID。支持平台的完整列表见[**此处**](../../docs/mapping.md)。                                                          |
| `--output-dir`            | 生成的 `.adla` 模型的保存目录。可选，默认为 `../model`。 |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"` 时（如 `w8a16 (f16)` 或 `w16a16 (f16)`）**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置。
>
> 若未提供数据集，AMLNN 会生成随机校准数据而不是报错，这可能导致模型精度严重下降。
>
> `activation_dtype` 不适用于 `w8a8`。

转换完成后，预期的模型路径为：

```text
examples/blazepose_detect/model/pose_detection_float32_w8a8.adla
```

## 3. 运行 Python 示例

**前置条件：**
- Python 3.10
- 依赖包：`amlnn`

**安装依赖：**
```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

**在设备上运行：**
```bash
python blazepose_detect_inference.py \
    --adla ../model/pose_detection_float32_w8a8.adla \
    --image-dir ../input \
    --anchor ./anchors.npy \
    --conf 0.5 \
    --nms 0.3
```
参数说明：

| 参数              | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| `--adla` | 编译后的 `.adla` 格式检测模型的路径。 |
| `--image-dir` | 待处理测试图像所在目录的路径。 |
| `--anchor` | （可选）包含检测器 anchor box 坐标的 NumPy 文件路径。默认为 `./anchors.npy`。 |
| `--conf` | （可选）置信度分数阈值，仅保留分数高于该值的检测结果。默认为 `0.5`。 |
| `--nms` | （可选）非极大值抑制过程中使用的交并比（IoU）阈值，用于滤除重叠的边界框。默认为 `0.3`。 |

脚本会自动处理当前目录下的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将结果保存到 `{model_name}_result` 文件夹。


## 4. 运行 C++ 示例

### Android 构建

#### 前置条件

* 已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 预编译的 Android OpenCV 位于 `dependency/opencv/` 文件夹。

#### 1. 配置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/blazepose_detect/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在对应所选 Android ABI 的构建目录中：

* 64 位：`build/android/arm64-v8a/blazepose_detect_demo`
* 32 位：`build/android/armeabi-v7a/blazepose_detect_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/blazepose_detect_demo /data/local/tmp/
adb push ../model/pose_detection_float32_w8a8.adla /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x blazepose_detect_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./blazepose_detect_demo <model_path> <image_dir>
./blazepose_detect_demo pose_detection_float32_w8a8.adla input/
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/blazepose_detect_demo` 及对应的 32 位库路径。将 `pose_detection_float32_w8a8.adla` 替换为实际的模型文件名。

---

### Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 目录必须已加入系统的 `PATH`。
* 预编译的 OpenCV 位于 `dependency/opencv/` 文件夹。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 配置环境

将下载好的工具链加入 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本找到编译器和神经网络依赖。

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
cd examples/blazepose_detect/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 若编译器的前缀名称不同（例如用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在对应所选架构的构建目录中：

* 64 位：`build/linux/64/blazepose_detect_demo`
* 32 位：`build/linux/32/blazepose_detect_demo`

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
cd examples/blazepose_detect/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而无需传入 `-s`、`-t` 参数。

可执行文件将生成在对应所选架构的构建目录中：

* 64 位：`build/yocto/64/blazepose_detect_demo`
* 32 位：`build/yocto/32/blazepose_detect_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/blazepose_detect_demo /data/local/tmp/
adb push ../model/pose_detection_float32_w8a8.adla /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x blazepose_detect_demo

# Usage: ./blazepose_detect_demo <model_path> <image_dir>
./blazepose_detect_demo pose_detection_float32_w8a8.adla input/
```

> **注意：** 将 `pose_detection_float32_w8a8.adla` 替换为实际的模型文件名。若使用 32 位或 Yocto 构建，请调整可执行文件路径。


## 5. 结果

### 性能信息

将日志级别设为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统与 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟与 DRAM 带宽占用。

### 检测输出

对每张输入图像，示例会在模型结果目录中生成两个输出文件：

1. **检测数据**
   保存为：

   ```text
   <model_name>_result/<image_name>_det.txt
   ```

   该 `.txt` 文件包含检测到的每个姿态所对应的 BlazePose 检测器输出值。

2. **可视化图像**
   保存为：

   ```text
   <model_name>_result/<image_name>_result.jpg
   ```

   结果图像包含检测到的姿态边界框及可视化输出。

控制台还会打印每张图像检测到的姿态数量。

例如：

```text
pose_detection_float32_w8a8_result/
├── test_image_det.txt
└── test_image_result.jpg
```

可将结果文件从设备拉回本地查看：

```bash
adb pull pose_detection_float32_w8a8_result/test_image_result.jpg
adb pull pose_detection_float32_w8a8_result/test_image_det.txt
```

![BlazePose 检测结果](./result.jpg)

> **注意：** 检测器的输出将作为 BlazePose 关键点模型的输入，用于后续的姿态关键点估计。



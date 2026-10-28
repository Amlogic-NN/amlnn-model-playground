[EN](README.md) | **CN**

# PaddleOCR 文字识别 v4

本示例使用 AMLNN 运行 PPOCR_Rec。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看文字识别结果。

## 目录结构

```bash
examples/ppocr_rec_v4/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images and recognition dictionary
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── result.jpg         # Example result
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本项目分发 PP-OCR 识别 v4 模型，该模型最初由百度（PaddlePaddle）开发，取自 PaddleOCR 官方仓库。模型权重和网络结构依据宽松的 Apache 2.0 许可证提供。如果你进一步再分发这些模型文件或将其用于商业用途，请务必保留 Apache 2.0 许可证声明。

本示例中的模型预处理和后处理代码部分由 AI 生成。如果其中某部分与已有工作相似并引起疑虑，请联系我们，我们会将其删除或调整。

## 1. 准备 ONNX 模型

### 下载 ONNX

下载已准备好的 ONNX 模型：

- [点击此处下载 PaddleOCR 文字识别 v4 ONNX 模型！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/ppocr_rec_v4/ppocr_rec_static.onnx)

将下载的模型放到：

```text
examples/ppocr_rec_v4/model/ppocr_rec_static.onnx
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/ppocr_rec_v4/py` 目录下运行 ADLA 导出脚本：

```bash
cd examples/ppocr_rec_v4/py
python export_adla.py \
  --onnx ../model/ppocr_rec_static.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                | 说明                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | PaddleOCR 文字识别 v4 `.onnx` 模型的路径。                                                      |
| `--target-platform` | 目标平台 ID。完整的目标平台列表请见[**此处**](../../docs/mapping.md)。                          |
| `--output-dir`      | （可选）生成的 `.adla` 模型的保存目录。默认为 `../model`。                                      |

预期的模型路径：

```text
examples/ppocr_rec_v4/model/ppocr_rec_static_w8a16.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需的软件包：`numpy`、`opencv-python`、`amlnn`

### 安装依赖

```bash
pip install numpy opencv-python amlnn_edge_toolkit-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

```bash
python ppocr_rec_inference.py \
  --adla ../model/ppocr_rec_static_w8a16.adla \
  --dict ../input/ppocr_keys_v1.txt \
  --image-dir ../input
```

参数说明：

| 参数          | 说明                                                   |
| ------------- | ------------------------------------------------------ |
| `--adla`      | PaddleOCR 文字识别 v4 `.adla` 模型的路径。             |
| `--dict` | PP-OCR 识别字典 `.txt` 文件的路径。                    |
| `--image-dir` | 存放测试图像的目录。                                   |

脚本会自动处理指定图像目录中所有受支持的图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将识别出的文字打印到控制台。

## 4. 运行 C++ 示例

### Android 构建

#### 前置条件

* 已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 预编译的 OpenCV 位于 `dependency/opencv/` 目录下。

#### 1. 设置环境

导出 NDK 和 AMLNN 的路径，以便构建脚本能够找到所需的工具链和神经网络依赖。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

```bash
cd examples/ppocr_rec_v4/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/ppocrv4_rec_demo`
* 32 位：`build/android/armeabi-v7a/ppocrv4_rec_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/ppocrv4_rec_demo /data/local/tmp/
adb push ../model/ppocr_rec_static_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x ppocrv4_rec_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./ppocrv4_rec_demo <model_path> <image_dir> <dict_path>
./ppocrv4_rec_demo ppocr_rec_static_w8a16.adla input/ input/ppocr_keys_v1.txt
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/ppocrv4_rec_demo` 以及对应的 32 位库路径。请将 `ppocr_rec_static_w8a16.adla` 替换为你实际生成的模型文件名。

---

### Linux 构建

Linux 构建流程支持两种不同的模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译器工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 目录必须添加到系统的 `PATH` 中。
* 预编译的 OpenCV 位于 `dependency/opencv/` 目录下。
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
cd examples/ppocr_rec_v4/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名称不同，例如使用 `aarch64-linux-gnu` 而不是 `aarch64-none-linux-gnu`，可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/ppocrv4_rec_demo`
* 32 位：`build/linux/32/ppocrv4_rec_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已具备 CMake 工具链文件。
* 预编译的 OpenCV 位于 `dependency/opencv/` 目录下。

##### 2. 设置环境

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/ppocr_rec_v4/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不必传入 `-s` 和 `-t` 标志。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/ppocrv4_rec_demo`
* 32 位：`build/yocto/32/ppocrv4_rec_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/ppocrv4_rec_demo /data/local/tmp/
adb push ../model/ppocr_rec_static_w8a16.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x ppocrv4_rec_demo

# Usage: ./ppocrv4_rec_demo <model_path> <image_dir> <dict_path>
./ppocrv4_rec_demo ppocr_rec_static_w8a16.adla input/ input/ppocr_keys_v1.txt
```

> **注意：** 请将 `ppocr_rec_static_w8a16.adla` 替换为你实际生成的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理结束后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延和 DRAM 带宽占用。

### 识别输出

对每张输入图像，程序会执行文字识别，并将识别出的文字和推理时间打印到控制台。

示例输出：

![替代文本](./result.jpg)
```text
============================================================
Processing image: "test.jpg"
============================================================
Inference time: 7.22225 ms
[RESULT] Recognized Text: Hello 123 发 abc idk
============================================================
```

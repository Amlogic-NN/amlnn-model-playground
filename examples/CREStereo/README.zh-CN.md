[EN](README.md) | **CN**

# CREStereo 示例

本示例使用 AMLNN 运行 CREStereo。完整流程如下：

1. 下载预先准备好的 ADLA 模型。
2. 使用该 ADLA 模型运行 Python 示例。
3. 使用该 ADLA 模型运行 C++（Linux/Android）示例。
4. 查看视差结果。

CREStereo 以一对左右立体图像作为输入，生成可用于立体深度估计的视差图。

## 目录结构

```bash
examples/CREstereo/
├── cpp/               # C++ demo and build scripts
├── input/             # Left and right stereo images
├── model/             # Put the ADLA model here
├── py/                # Python inference demo
└── result.png         # Example disparity output
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例使用的 CREStereo 模型由第三方提供，最初来自旷视研究院（Megvii Research）的官方 CREStereo 仓库，该仓库以 Apache License 2.0 发布。为配合本示例，我们分发的是经过修改并编译的 `.adla` 版本模型。在再分发该模型或其修改版本时（包括商业用途），请保留所需的版权与许可证声明，并遵循原始 Apache License 2.0 条款。

本示例中的模型预处理和后处理代码部分由 AI 生成。如果任何部分无意中与已有工作相似并引起疑虑，请联系我们，我们会及时删除或调整。

## 1. 准备 ADLA 模型

### 下载 ADLA

下载预先准备好的 ADLA 模型，并将其放到 `examples/CREstereo/model/` 目录下：

- [在此下载 CREStereo ADLA 文件！](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/CREstereo/crestereo_combined_iter2_240x320_sim.adla)

期望路径：

```text
examples/CREstereo/model/crestereo_combined_iter2_240x320_sim.adla
```

## 2. 运行 Python 示例

### 前置条件

* Python 3.10
* 所需依赖包：`numpy`、`opencv-python`、`amlnn`

### 安装依赖

```bash
pip install numpy opencv-python amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

### 在设备上运行

```bash
python crestereo_inference.py \
    --adla ../model/crestereo_combined_iter2_240x320_sim.adla \
    --left ../input/left.png \
    --right ../input/right.png
```

参数说明：

| 参数      | 说明                                                              |
| --------- | ------------------------------------------------------------------------ |
| `--adla`  | 编译后的 CREStereo 模型路径，格式为 `.adla`。                  |
| `--left`  | 左侧立体图像路径。  |
| `--right` | 右侧立体图像路径。 |

左图和右图构成一对立体图像，应来自同一场景。

## 3. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 系统中已安装 **Android NDK**（推荐 r27d）。
* 已下载并解压 **AMLNN Toolkit**。
* 位于 `dependency/opencv/` 文件夹中的 Android 预编译 OpenCV。

#### 1. 设置环境

导出 NDK 和 AMLNN 的路径，以便构建脚本找到所需的工具链和依赖。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/CREstereo/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的构建目录中：

* 64 位：`build/android/arm64-v8a/crestereo_demo`
* 32 位：`build/android/armeabi-v7a/crestereo_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/crestereo_demo /data/local/tmp/
adb push ../model/crestereo_combined_iter2_240x320_sim.adla /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x crestereo_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./crestereo_demo <model.adla> <left_image> <right_image>
./crestereo_demo crestereo_combined_iter2_240x320_sim.adla input/left.png input/right.png
```

> **注意：** 对于 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/crestereo_demo` 及对应的 32 位库路径。请将 `crestereo_combined_iter2_240x320_sim.adla` 替换为实际的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* GCC 交叉编译工具链（推荐 GCC 10.3）。
* 工具链的 `bin/` 文件夹必须添加到系统的 `PATH` 中。
* 位于 `dependency/opencv/` 文件夹中的预编译 OpenCV。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

将下载的工具链添加到 `PATH`，并导出 `AMLNN_HOME` 变量。

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
cd examples/CREstereo/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名不同（例如使用 `aarch64-linux-gnu` 而不是 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量来覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/linux/64/crestereo_demo`
* 32 位：`build/linux/32/crestereo_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* CMake 工具链文件可用。
* 位于 `../../../dependency/opencv/`（相对于脚本目录）的预编译 OpenCV。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/CREstereo/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不必传入 `-s` 和 `-t` 参数。

可执行文件将生成在所选架构对应的构建目录中：

* 64 位：`build/yocto/64/crestereo_demo`
* 32 位：`build/yocto/32/crestereo_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/crestereo_demo /data/local/tmp/
adb push ../model/crestereo_combined_iter2_240x320_sim.adla /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x crestereo_demo

# Usage: ./crestereo_demo <model.adla> <left_image> <right_image>
./crestereo_demo crestereo_combined_iter2_240x320_sim.adla input/left.png input/right.png
```

> **注意：** 请将 `crestereo_combined_iter2_240x320_sim.adla` 替换为实际的模型文件名。如果使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 4. 结果

### 性能信息

将日志级别设置为 `INFO` 后，程序会在推理完成后输出运行时性能信息。控制台输出可能包括：

* **硬件信息：** 系统和 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理延迟和 DRAM 带宽占用。

### 视差输出

CREStereo 示例处理左右立体图像，并生成表示两个视角之间估计像素位移的视差图。

生成的视差图会保存为视差图像以便可视化。

![CREStereo 视差结果](./result.png)

> **注意：** 视差图表示左右图像之间的立体对应关系，可用于深度估计。

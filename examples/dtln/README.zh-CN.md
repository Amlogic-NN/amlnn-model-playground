[EN](README.md) | **CN**

# DTLN 示例

本示例演示如何使用 AMLNN 运行 [DTLN](https://github.com/breizhn/DTLN) 语音增强模型。

完整流程如下：

1. 下载或准备 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 格式。
3. 使用 ADLA 模型构建并运行 C++ 示例。
4. 获取增强后的输出音频。

## 目录结构

```text
examples/dtln/
|-- cpp/                  # C++ demo source and build scripts
|-- model/                # Place ONNX and ADLA models here
|-- in.wav                # Example input audio
|-- out.wav               # Example output audio
```

## 许可证

本示例代码基于 Apache License 2.0 许可发布。详见仓库根目录的 [LICENSE](../../LICENSE) 文件。

## 1. 准备 ONNX 模型

DTLN 示例使用两个预训练的 ONNX 模型：

- [model_1.onnx](https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_1.onnx)
- [model_2.onnx](https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_2.onnx)

将这两个文件下载到 `examples/dtln/model/` 目录。

## 2. 将 ONNX 转换为 ADLA

使用提供的转换脚本从 ONNX 文件生成 ADLA 模型。

```bash
python ./py/export_adla.py --model1 ./model/model_1.onnx --model2 ./model/model_2.onnx --target-platform 007
```

转换完成后，将生成的 ADLA 文件放到 `examples/dtln/model/` 目录下。

## 3. 运行 Python 示例

## 4. 构建 C++ 示例

### 为 Android 构建

**前置条件：**

- Android NDK（推荐 r25e）

**构建：**

在 `examples/dtln/cpp/` 目录下构建 Android 示例。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25e
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/dtln/cpp
./build-android.sh
```

生成的可执行文件位于 `examples/dtln/cpp/build/android/dtln_demo`。

**运行：**

将示例可执行文件、模型文件和输入音频拷贝到设备上。

```bash
adb push examples/dtln/cpp/build/android/dtln_demo /data/local/tmp/
adb push examples/dtln/model/dtln_model_1.adla examples/dtln/model/dtln_model_2.adla /data/local/tmp/
adb push examples/dtln/in.wav /data/local/tmp/
```

在设备上运行示例：

```bash
adb shell
cd /data/local/tmp
chmod +x dtln_demo
export LD_LIBRARY_PATH=/vendor/lib64
./dtln_demo ./dtln_model_1.adla ./dtln_model_2.adla ./in.wav ./out.wav
```

### 为 Linux 构建

Linux 构建流程支持两种不同的模式：**标准 Linux 交叉编译**（默认）和 **Yocto SDK 编译**。

### 模式 1：标准 Linux 交叉编译（默认）

**前置条件：**

- GCC 交叉编译工具链（推荐 GCC 10.3）。
- 工具链的 `bin/` 目录必须已加入系统的 `PATH`。
- 已设置 `AMLNN_HOME` 环境变量

**配置环境：**

将下载好的工具链加入 `PATH`，并导出 `AMLNN_HOME` 变量，以便脚本找到编译器和神经网络依赖。

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds, add the 64-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# OR for 32-bit (arm) builds, add the 32-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

**构建：**

```bash
cd examples/dtln/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

*（可选覆盖）：* 若编译器的前缀名称不同（例如用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在 `build/linux/64/dtln_demo`（或 `build/linux/32/dtln_demo`）。

### 模式 2：Yocto 构建

**前置条件：**

- 已安装 Yocto SDK
- 已有可用的 CMake Toolchain 文件

**构建：**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/dtln/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Or build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

*（注意：也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不传 `-s` 和 `-t` 参数。）*

可执行文件将生成在 `build/yocto/64/dtln_demo`（或 `build/yocto/32/dtln_demo`）。

---

**运行：**

运行流程与上文 Android 部分完全相同。

## 5. 结果

示例会将 `out.wav` 写入设备目录。

结果示例：

- [in.wav](./in.wav)
- [out.wav](./out.wav)

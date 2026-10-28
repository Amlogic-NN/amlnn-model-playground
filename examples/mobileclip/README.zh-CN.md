[EN](README.md) | **CN**

# MobileCLIP 示例

本示例演示在 Amlogic NPU 平台上使用 AMLNN 进行图文匹配，包含针对视觉编码器和文本编码器 `.adla` 模型的 Python 与 C++ 推理示例。

## 目录结构

```bash
examples/mobileclip/
├── cpp/                    # C++ demo and build scripts (Android / Linux)
│   └── src/                # mobileclip_demo source
├── input/                  # Demo input assets
├── model/                  # Place runtime .adla models here
└── py/                     # Python inference scripts
    ├── clip_tokenizer.py
    └── mobileclip_inference.py
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本示例中的模型预处理和后处理代码部分由 AI 生成。若某些内容与已有工作相似并引起顾虑，请联系我们，我们会予以删除或调整。

## 1. 前置条件

将视觉编码器和文本编码器的 `.adla` 模型放到 `model/` 目录下。

模型输入输出约定：


| 编码器 | 输入名称   | 输入形状           |
| ------- | ---------- | ------------------ |
| 视觉    | `image`    | `[1, 3, 256, 256]` |
| 文本    | `text`     | `[1, 77]`          |


## 2. 运行 Python 示例

**前置条件：**

- Python 3.10+
- 所需依赖包：`numpy`、`Pillow`、`amlnn`
- 一个包含 `vocab.json` 和 `merges.txt` 的 tokenizer 目录

**安装依赖示例：**

```bash
pip install numpy Pillow amlnn_toolkit-1.0.0-cp312-cp312-linux_aarch64.whl
```

**在设备上运行：**

```bash
cd py

python mobileclip_inference.py \
  --vision-model ../model/vision.adla \
  --text-model ../model/text.adla \
  --tokenizer-dir /path/to/tokenizer \
  --image-path ../input/test.jpg \
  --texts "a red bus" "a red handbag" "a blue jacket"
```


| 参数              | 说明                                                                    |
| ----------------- | ----------------------------------------------------------------------- |
| `--vision-model`  | 视觉编码器 `.adla` 模型的路径                                           |
| `--text-model`    | 文本编码器 `.adla` 模型的路径                                           |
| `--tokenizer-dir` | 包含 `vocab.json` 和 `merges.txt` 的目录                                |
| `--image-path`    | 可选的输入图像路径。若省略，脚本会以交互方式提示输入                    |
| `--texts`         | 可选的要比较的文本描述                                                  |
| `--logit-scale`   | 可选的 logit 缩放系数，默认为 `100.0`                                   |
| `--runtime-mode`  | `native` 或 `nnserver`，默认为 `native`                                 |
| `--enable-perf`   | 启用 AMLNN 性能采集                                                     |


如果省略 `--image-path` 和 `--texts`，脚本会进入与 C++ 示例类似的交互模式。

## 3. 运行 C++ 示例

### 为 Android 构建

**前置条件：**

- Android NDK（建议 r27d）
- 已下载并解压 AMLNN toolkit

**1. 设置环境：**

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

**2. 构建：**

```bash
cd examples/mobileclip/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在：

```text
build/android_arm64-v8a/mobileclip_demo
```

**运行：**

```bash
adb shell "mkdir -p /data/mobileclip"
adb push build/android_arm64-v8a/mobileclip_demo /data/mobileclip/
adb push ../model/vision.adla /data/mobileclip/models/
adb push ../model/text.adla /data/mobileclip/models/
adb push /path/to/tokenizer /data/mobileclip/tokenizer/
adb push ../input/000000004505.jpg /data/mobileclip/input/

adb shell
cd /data/mobileclip
chmod +x mobileclip_demo

# Usage:
# mobileclip_demo [options] <vision_model.adla> <text_model.adla> <tokenizer_dir> <image.jpg>
./mobileclip_demo \
  models/vision.adla \
  models/text.adla \
  tokenizer/ \
  input/000000004505.jpg
```

可选参数：


| 参数                  | 说明                                                     |
| --------------------- | -------------------------------------------------------- |
| `--neon true/false`   | 启用/禁用 ADLA soft ops 的 NEON（默认：true）            |
| `--openmp true/false` | 启用/禁用 ADLA soft ops 的 OpenMP（默认：false）         |
| `--openmp-num N`      | OpenMP 线程数，默认为 `2`                                |
| `--profiling`         | 启用性能分析输出                                         |


### 为 Linux 构建

Linux 构建支持两种模式：**标准交叉编译**（默认）与 **Yocto SDK 编译**。

#### 模式 1：标准 Linux 交叉编译（默认）

**前置条件：**

- 一套 GCC 交叉编译器工具链（建议 GCC 10.3+）
- 已将工具链的 `bin/` 目录加入 `PATH`
- 已设置 `AMLNN_HOME` 环境变量

**1. 设置环境：**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# For 32-bit (arm) builds:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

**2. 构建：**

```bash
cd examples/mobileclip/cpp

# Build for 64-bit (default)
./build-linux.sh

# Or explicitly:
./build-linux.sh -b 64
./build-linux.sh -a aarch64

# Build for 32-bit
./build-linux.sh -b 32
./build-linux.sh -a armhf
```

如果你的编译器前缀不同，可通过 `GCC_COMPILER` 覆盖：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在：

```text
build/linux/64/mobileclip_demo
build/linux/32/mobileclip_demo
```

#### 模式 2：Yocto/Debian/Armbian 构建

**前置条件：**

- 已安装 Yocto SDK
- 已备有 CMake 工具链文件

**构建：**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/mobileclip/cpp

# Build for Yocto 64-bit (default)
./build-linux.sh -m yocto -b 64 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

也可以设置 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE`，而不通过 `-s` 和 `-t` 传入。

可执行文件将生成在：

```text
build/yocto/64/mobileclip_demo
build/yocto/32/mobileclip_demo
```

**在 Linux 设备上运行：**

将可执行文件、模型、tokenizer 目录和输入图像复制到目标设备，然后运行：

```bash
./mobileclip_demo \
  models/vision.adla \
  models/text.adla \
  tokenizer/ \
  input/000000004505.jpg
```

可选参数与 Android 示例相同（`--neon`、`--openmp`、`--openmp-num`、`--profiling`）。

## 4. 结果

**示例：**

```bash
$ ./mobileclip_demo \
    models/vision.adla \
    models/text.adla \
    tokenizer/ \
    input/000000004505.jpg

[Info] Loading tokenizer from: tokenizer/
[Info] Initializing vision model: models/vision.adla
[Info] Initializing text model: models/text.adla
[Info] Models initialized successfully.

Enter text descriptions (comma-separated, or 'skip' for defaults):
> a red bus, a red handbag, a blue jacket

============================================================
MobileCLIP-S2 Image-Text Matching Results
============================================================
Image: input/000000004505.jpg
------------------------------------------------------------
[1] prob=1.000000  sim=0.295992  text='a red bus'
[2] prob=0.000000  sim=0.147470  text='a red handbag'
[3] prob=0.000000  sim=0.111335  text='a blue jacket'
============================================================
```

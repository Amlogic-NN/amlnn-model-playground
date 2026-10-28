[EN](README.md) | **CN**

# Clip 示例

本示例使用 AMLNN 运行 Clip。完整流程如下：

1. 准备或下载 ONNX 模型。
2. 将 ONNX 模型转换为 ADLA 模型。
3. 使用 ADLA 模型运行 Python 示例。
4. 使用 ADLA 模型运行 C++（Linux/Android）示例。
5. 查看图文匹配结果。

## 目录结构

```bash
examples/clip/
├── cpp/               # C++ demo and build scripts
├── input/             # Input images for demo
├── model/             # Put ONNX and ADLA models here
├── py/                # Python conversion and demo scripts
└── tokenizer/         # Tokenizer directory
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。详情请参阅仓库根目录的 [LICENSE](../../LICENSE) 文件。

本项目分发的 CLIP ONNX 模型最初由 OpenAI 开发，取自 Hugging Face。模型权重与网络结构按宽松的 MIT 许可证提供。如果你要进一步再分发这些模型文件或将其用于商业用途，请务必保留 MIT 许可证声明。

> **MIT 许可证**
> 
> Copyright (c) 2021 OpenAI
> 
> 特此免费授予任何获得本软件及相关文档文件（以下称“本软件”）副本的人不受限制地处置本软件的权利，
> 包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售本软件副本的权利，
> 并允许本软件的受赠人这样做，但须满足以下条件：
> 
> 上述版权声明和本许可声明应包含在本软件的所有副本或主要部分中。
> 
> 本软件按“原样”提供，不附带任何明示或暗示的保证，包括但不限于对适销性、
> 特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、
> 损害或其他责任承担责任，无论是合同行为、侵权行为还是其他行为，亦无论其是否
> 因本软件或本软件的使用或其他交易而产生。

本示例中的模型预处理和后处理代码部分由 AI 生成。若有内容无意中与已有工作相似并引起顾虑，请联系我们，我们会及时删除或调整。


## 1. 准备 ONNX 模型

### 下载模型

下载准备好的模型和 tokenizer：

### [文本 ONNX 模型](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/clip/text_model.onnx)
### [图像 ONNX 模型](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/clip/vision_model.onnx)
### [CLIP Tokenizer 压缩包](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/clip/clip_tokenizer.zip)

将 ONNX 模型放到 `examples/clip/model/` 下，并将 tokenizer 解压到 `examples/clip/tokenizer/` 下。

期望路径：

```text
examples/clip/model/text_model.onnx
examples/clip/model/vision_model.onnx
examples/clip/tokenizer/
```

## 2. 将 ONNX 转换为 ADLA

在 `examples/clip/py` 下运行 ADLA 导出脚本：

```bash
cd examples/clip/py
python export_adla.py \
  --text-onnx ../model/text_model.onnx \
  --image-onnx ../model/vision_model.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| 参数                    | 说明 |
| ----------------------- | ---- |
| `--text-onnx`           | 文本编码器 `.onnx` 模型的路径。 |
| `--image-onnx`          | 图像编码器 `.onnx` 模型的路径。 |
| `--text-dataset-path`   | （可选）包含文本模型量化校准所需输入路径的 `.txt` 文件路径。`w8a8` 和 `w8a16 (i16)` 必填。[1] |
| `--image-dataset-path`  | （可选）包含图像模型量化校准所需输入路径的 `.txt` 文件路径。`w8a8` 和 `w8a16 (i16)` 必填。[1] |
| `--target-platform`     | 目标平台 ID。支持的平台完整列表见[**此处**](../../docs/mapping.md)。 |
| `--output-dir`          | （可选）生成的 `.adla` 模型的保存目录，默认为 `../model`。 |

> **[1] 量化数据集说明**
>
> 使用 `activation_dtype="f16"`（例如 `w8a16 (f16)` 或 `w16a16 (f16)`）时**不需要**校准数据集。请在 `amlnn.config()` 调用中显式设置该项。
>
> 如果未提供数据集，AMLNN 会生成随机校准数据而不会报错，这可能导致模型精度严重下降。
>
> `activation_dtype` 不适用于 `w8a8`。

> **注意：** 建议两个模型使用相同的量化方法，因为图像和文本的特征向量输出会直接比较。

在当前的 `w8a16 (f16)` 配置下，期望的模型路径为：

```text
examples/clip/model/text_model_w8a16.adla
examples/clip/model/vision_model_w8a16.adla
```

## 3. 运行 Python 示例

### 前置条件

* Python 3.10
* 依赖包：`numpy`、`opencv-python`、`amlnn`、`transformers`

### 安装依赖

```bash
pip install numpy opencv-python amlnn_edge_toolkit-1.0.0-cp310-cp310-linux_aarch64.whl transformers
```

### 在设备上运行

```bash
python clip_inference.py \
    --image ../model/vision_model_w8a16.adla \
    --text ../model/text_model_w8a16.adla \
    --tokenizer ../tokenizer/ \
    --image-dir ../input \
    --prompts "a red bus" "apple" "car" "person" \
    --logit-scale 100.0
```

参数说明：

| 参数            | 说明 |
| --------------- | ---- |
| `--image`       | 图像编码器 `.adla` 模型的路径。 |
| `--text`        | 文本编码器 `.adla` 模型的路径。 |
| `--tokenizer`   | CLIPTokenizer 目录路径。 |
| `--image-dir`   | 存放测试图像的目录。 |
| `--prompts`     | 用于与每张图像比较的文本提示词。多个提示词之间用空格分隔，由多个单词组成的提示词需用引号包裹。 |
| `--logit-scale` | （可选）用于图文相似度得分的 logit 缩放系数，默认为 `100.0`。 |

脚本会自动处理指定图像目录下的所有图像文件（`.jpg`、`.jpeg`、`.png`、`.bmp`），并将结果保存到 `{model_name}_result` 文件夹。

## 4. 运行 C++ 示例

### 为 Android 构建

#### 前置条件

* 已在系统中安装 **Android NDK**（建议 r27d）。
* 已下载并解压 **AMLNN Toolkit**。

#### 1. 设置环境

导出 NDK（工具链）和 AMLNN（神经网络依赖）的路径，以便脚本能够找到它们。

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. 构建

进入 C++ 目录并运行构建脚本。

```bash
cd examples/clip/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

可执行文件将生成在所选 Android ABI 对应的 build 目录中：

* 64 位：`build/android/arm64-v8a/clip_demo`
* 32 位：`build/android/armeabi-v7a/clip_demo`

#### 3. 运行示例

以下示例使用默认的 64 位（`arm64-v8a`）构建。

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/clip_demo /data/local/tmp/
adb push ../model/vision_model_w8a16.adla /data/local/tmp/
adb push ../model/text_model_w8a16.adla /data/local/tmp/
adb push ../tokenizer /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x clip_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./clip_demo <image_model> <text_model> <tokenizer_dir> <image_dir> [texts ...]
./clip_demo vision_model_w8a16.adla text_model_w8a16.adla tokenizer/ input/ "a red bus" "apple" "car" "person"
```

> **注意：** 若使用 32 位（`armeabi-v7a`）构建，请使用 `build/android/armeabi-v7a/clip_demo` 及对应的 32 位库路径。请将 `.adla` 模型文件名替换为你实际生成的模型文件名。

---

### 为 Linux 构建

Linux 构建流程支持两种不同模式：

1. **标准 Linux 交叉编译**（默认）
2. **Yocto SDK 编译**

#### 模式 1：标准 Linux 交叉编译（默认）

##### 前置条件

* 一套 GCC 交叉编译器工具链（建议 GCC 10.3）。
* 工具链的 `bin/` 目录必须加入系统的 `PATH`。
* 已设置 `AMLNN_HOME` 环境变量。

##### 1. 设置环境

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
cd examples/clip/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **可选覆盖：** 如果你的编译器前缀名不同（例如使用 `aarch64-linux-gnu` 而非 `aarch64-none-linux-gnu`），可以通过设置 `GCC_COMPILER` 变量覆盖默认值：

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

可执行文件将生成在所选架构对应的 build 目录中：

* 64 位：`build/linux/64/clip_demo`
* 32 位：`build/linux/32/clip_demo`

#### 模式 2：Yocto/Debian/Armbian 构建

##### 1. 前置条件

* 已安装 Yocto SDK。
* 已有可用的 CMake Toolchain 文件。

##### 2. 设置环境

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. 构建

```bash
cd examples/clip/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **注意：** 也可以使用 `YOCTO_SDK_ROOT` 和 `TOOLCHAIN_FILE` 环境变量，而不通过 `-s` 和 `-t` 参数传入。

可执行文件将生成在所选架构对应的 build 目录中：

* 64 位：`build/yocto/64/clip_demo`
* 32 位：`build/yocto/32/clip_demo`

#### 3. 运行示例

以下示例使用默认的 64 位 Linux 构建。

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/clip_demo /data/local/tmp/
adb push ../model/vision_model_w8a16.adla /data/local/tmp/
adb push ../model/text_model_w8a16.adla /data/local/tmp/
adb push ../tokenizer /data/local/tmp/
adb push ../input /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x clip_demo

# Usage: ./clip_demo <image_model> <text_model> <tokenizer_dir> <image_dir> [texts ...]
./clip_demo vision_model_w8a16.adla text_model_w8a16.adla tokenizer/ input/ "a red bus" "apple" "car" "person"
```

> **注意：** 请将 `.adla` 模型文件名替换为你实际生成的模型文件名。若使用 32 位或 Yocto 构建，请相应调整可执行文件路径。

## 5. 结果

### 性能反馈

将日志级别设置为 `INFO` 后，程序会在推理完成后输出运行时性能信息。控制台输出可能包含：

* **硬件信息：** 系统版本与 ADLA 库版本。
* **模型概览：** 输入和输出张量配置。
* **NPU 指标：** 推理时延与 DRAM 带宽占用。

### 图文匹配输出

对每张输入图像，示例会将归一化后的图像特征向量与给定的文本特征向量进行比较，并按概率排序打印匹配结果。

示例：

```text
============================================================
Processing image 1/1: 000000004505.jpg
============================================================
I amlnn_inference_base amlnn_inference_breakdown_ms: input_prepare=0.026, runtime_inference=11.382, output_postprocess=0.024, total=11.436
Image embedding shape: (1, 512)
  1. probability=0.995906, similarity=0.286593, text='a photo of a bus'
  2. probability=0.001184, similarity=0.219244, text='a photo of a car'
  3. probability=0.000906, similarity=0.216573, text='a photo of a truck'
  4. probability=0.000903, similarity=0.216530, text='a photo of a person'
  5. probability=0.000622, similarity=0.212815, text='a photo of a train'
  6. probability=0.000185, similarity=0.200700, text='a photo of a bicycle'
  7. probability=0.000129, similarity=0.197068, text='a photo of a bird'
  8. probability=0.000109, similarity=0.195386, text='a photo of a dog'
  9. probability=0.000027, similarity=0.181607, text='a photo of a cat'
  10. probability=0.000026, similarity=0.181202, text='a photo of a motorcycle'
  11. probability=0.000002, similarity=0.156530, text='a photo of a airplane'

============================================================
```

结果显示每个给定文本提示词对应的概率和余弦相似度得分，并按图文匹配程度从高到低排序。

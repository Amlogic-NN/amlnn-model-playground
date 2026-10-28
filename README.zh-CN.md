[EN](README.md) | **CN**

# Amlogic NN 模型库

<div align="left">
  <img src="poster1.jpg" width="100%" alt="Amlogic 技术横幅">
</div>

## 简介

**amlnn-model-playground** 基于 [amlnn-toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit) 开发。通过完成**模型转换**与**部署**步骤，我们为常用模型打造了开箱即用的**模型库**。示例包提供了完整的模型转换脚本，以及使用 **Python API** 和 **C API** 运行转换后模型的完整流程。

**目标：** 帮助用户快速上手并在 Amlogic NPU 平台上部署模型。模型库中丰富的算法库可指导开发者完成边缘 AI 产品的测试、基准测试、概念验证和部署。

## 依赖项

- **amlnn-runtime**：NPU 运行时库，参见 [amlnn-runtime](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_runtime)
- **amlnn-toolkit**：模型转换工具，参见 [amlnn-toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit)。
  - 安装包（`.whl`）路径：
    - [`amlnn_toolkit`](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit/whl)：用于 `amlnn_toolkit` / `amlnn_edge_toolkit` wheel（模型转换与运行时）
    - [`amlnn_toolkit_lite`](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit_lite/amlnn_edge_toolkit_lite/whl/aarch64)：用于 `amlnn_edge_toolkit_lite` wheel（仅模型运行时）
- **交叉编译工具链**：完整的工具链安装指南请参见[平台编译指南](docs/Platform_compile_guide.md)。

## 支持列表

### CNN

| 类别         | 链接                                                              | 模型名称                   | 量化数据类型 | 平台                              |
| ------------ | ----------------------------------------------------------------- | -------------------------- | ------------ | --------------------------------- |
| 分类         | [mobilenet](examples/mobilenet/README.zh-CN.md)                   | mobilenet_v2               | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 分类         | [resnet](examples/resnet/README.zh-CN.md)                         | resnet50-v2                | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 分类         | [DINO](examples/DINO/README.zh-CN.md)                             | DINO 骨干网络 + 线性分类器 | w8a16        | A311Y3/C305X2/A123X               |
| 音频分类     | [yamnet](examples/yamnet/README.zh-CN.md)                         | YAMNet                     | w8a16        | A311Y3/C305X2/A123X               |
| 目标检测     | [DETR](examples/DETR/README.zh-CN.md)                             | DETR                       | w16a16       | A311Y3/C305X2/A123X               |
| 目标检测     | [ppyoloe](examples/ppyoloe/README.zh-CN.md)                       | PP-YOLOE                   | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [qrcode](examples/qrcode/README.zh-CN.md)                         | 二维码检测                 | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yolov5](examples/yolov5/README.zh-CN.md)                         | YOLOv5                     | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yolov6](examples/yolov6/README.zh-CN.md)                         | YOLOv6                     | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yolov7](examples/yolov7/README.zh-CN.md)                         | YOLOv7                     | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yolov8](examples/yolov8/README.zh-CN.md)                         | YOLOv8                     | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yolov10](examples/yolov10/README.zh-CN.md)                       | YOLOv10                    | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yolov11](examples/yolov11/README.zh-CN.md)                       | YOLO11                     | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yoloe](examples/yoloe/README.zh-CN.md)                           | YOLOE                      | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yoloworld](examples/yoloworld/README.zh-CN.md)                   | YOLO-World                 | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 目标检测     | [yolox](examples/yolox/README.zh-CN.md)                           | YOLOX                      | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 人脸检测     | [retinaface](examples/retinaface/README.zh-CN.md)                 | RetinaFace                 | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 手势识别     | [gesture](examples/gesture/README.zh-CN.md)                       | Gesture                    | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 图像分割     | [deeplabv3](examples/deeplabv3/README.zh-CN.md)                   | DeepLabV3                  | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 图像分割     | [mobilesam](examples/mobilesam/README.zh-CN.md)                   | MobileSAM                  | w8a16        | A311Y3/C305X2/A123X               |
| 图像分割     | [ppseg](examples/ppseg/README.zh-CN.md)                           | PP-LiteSeg                 | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 实例分割     | [yolov5-seg](examples/yolov5-seg/README.zh-CN.md)                 | YOLOv5 分割                | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 实例分割     | [yolov8-seg](examples/yolov8-seg/README.zh-CN.md)                 | YOLOv8 分割                | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 旋转目标检测 | [yolov8-obb](examples/yolov8-obb/README.zh-CN.md)                 | YOLOv8 OBB                 | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 姿态估计     | [blazepose_detect](examples/blazepose_detect/README.zh-CN.md)     | BlazePose 检测             | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 姿态估计     | [blazepose_landmark](examples/blazepose_landmark/README.zh-CN.md) | BlazePose 关键点           | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 姿态估计     | [yolov8-pose](examples/yolov8-pose/README.zh-CN.md)               | YOLOv8 姿态                | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 车牌识别     | [lprnet](examples/lprnet/README.zh-CN.md)                         | LPRNet                     | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR 检测     | [ppocr-det](examples/ppocr-det/)                   | PaddleOCR 检测             | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR 检测     | [ppocr_det_v4](examples/ppocr_det_v4/README.zh-CN.md)             | PaddleOCR 检测 v4          | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR 检测     | [ppocrv5-det](examples/ppocrv5-det/README.zh-CN.md)               | PaddleOCR 检测 v5          | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR 识别     | [ppocr_rec_v4](examples/ppocr_rec_v4/README.zh-CN.md)             | PaddleOCR 识别 v4          | w8a16        | A311Y3/C305X2/A123X               |
| OCR 系统     | [ppocr_v4_system](examples/ppocr_v4_system/README.zh-CN.md)       | PaddleOCR v4 检测 + 识别   | w8a16        | A311Y3/C305X2/A123X               |
| OCR 系统     | [ppocrv5-system](examples/ppocrv5-system/README.zh-CN.md)         | PP-OCRv5 系统              | w8a16        | A311Y3/C305X2/A123X               |
| OCR 系统     | [ppocrv6-system](examples/ppocrv6-system/README.zh-CN.md)         | PP-OCRv6 系统              | w8a16        | A311Y3/C305X2/A123X               |
| 语音识别     | [sensevoice](examples/sensevoice/README.zh-CN.md)                 | SenseVoice                 | w8a16        | A311Y3/C305X2/A123X               |
| 语音识别     | [whisper](examples/whisper/README.zh-CN.md)                       | Whisper Tiny               | w8a8         | A311D2/S905X5/A311Y3/C305X2/A123X |
| 图文匹配     | [clip](examples/clip/README.zh-CN.md)                             | CLIP                       | w8a16        | A311Y3/C305X2/A123X               |
| 图文匹配     | [mobileclip](examples/mobileclip/README.zh-CN.md)                 | MobileCLIP                 | w8a16        | A311Y3/C305X2/A123X               |

### 大语言模型

| 模型名称          | 量化数据类型 | 平台                |
| ----------------- | ------------ | ------------------- |
| DeepSeek-R1-1.5B  | w4a16        | A311Y3/C305X2/A123X |
| Qwen-1.8B         | w4a16        | A311Y3/C305X2/A123X |
| Qwen1.5-0.5B      | w4a16        | A311Y3/C305X2/A123X |
| Qwen1.5-1.8B      | w4a16        | A311Y3/C305X2/A123X |
| Qwen2-0.5B        | w4a16        | A311Y3/C305X2/A123X |
| Qwen2-1.5B        | w4a16        | A311Y3/C305X2/A123X |
| Qwen2.5-0.5B      | w4a16        | A311Y3/C305X2/A123X |
| Qwen2.5-1.5B      | w4a16        | A311Y3/C305X2/A123X |
| Qwen2.5-3B        | w4a16        | A311Y3/C305X2/A123X |
| Qwen3-0.6B        | w4a16        | A311Y3/C305X2/A123X |
| Qwen3-1.7B        | w4a16        | A311Y3/C305X2/A123X |
| Qwen3-4B          | w4a16        | A311Y3/C305X2/A123X |
| TinyLlama0.4-1.1B | w4a16        | A311Y3/C305X2/A123X |
| TinyLlama1-1.1B   | w4a16        | A311Y3/C305X2/A123X |
| Llama2-7B         | w4a16        | A311Y3/C305X2/A123X |
| Llama3.2-1B       | w4a16        | A311Y3/C305X2/A123X |
| Llama3.2-3B       | w4a16        | A311Y3/C305X2/A123X |
| OpenLLaMA-3B      | w4a16        | A311Y3/C305X2/A123X |
| Gemma1-2B         | w4a16        | A311Y3/C305X2/A123X |
| Gemma2-2B         | w4a16        | A311Y3/C305X2/A123X |
| Gemma3-270M       | w4a16        | A311Y3/C305X2/A123X |
| Gemma3-1B         | w4a16        | A311Y3/C305X2/A123X |
| Gemma4-E2B        | w4a16        | A311Y3/C305X2/A123X |
| Gemma4-E4B        | w4a16        | A311Y3/C305X2/A123X |
| Phi1.5-1.3B       | w4a16        | A311Y3/C305X2/A123X |
| Phi2-2.7B         | w4a16        | A311Y3/C305X2/A123X |
| Phi3-3.8B         | w4a16        | A311Y3/C305X2/A123X |
| ChatGLM3-6B       | w4a16        | A311Y3/C305X2/A123X |
| MiniCPM3-4B       | w4a16        | A311Y3/C305X2/A123X |
| MiniCPM4-0.5B     | w4a16        | A311Y3/C305X2/A123X |
| InternVL2-1B      | w4a16        | A311Y3/C305X2/A123X |
| InternVL3-1B      | w4a16        | A311Y3/C305X2/A123X |
| Qwen-VL2-2B       | w4a16        | A311Y3/C305X2/A123X |
| Qwen-VL2.5-3B     | w4a16        | A311Y3/C305X2/A123X |
| Qwen-VL3-2B       | w4a16        | A311Y3/C305X2/A123X |
| Qwen-VL3-4B       | w4a16        | A311Y3/C305X2/A123X |
| MobileVLM-V2-1.7B | w4a16        | A311Y3/C305X2/A123X |
| MiniCPM-V2.6-8B   | w4a16        | A311Y3/C305X2/A123X |

**注意：** w4a16 表示权重使用 int4 量化，特征使用 float16。

## 性能

基准测试（FPS）与精度评估数据已移至独立的 [PERFORMANCE.md](PERFORMANCE.md) 页面。

## 示例编译

### AMLNN Runtime 环境准备

C++ 示例依赖 **AMLNN** 运行时库。构建系统会自动在同级目录中查找 `amlnn-toolkit`：

```
modelzoo/
├── amlnn-model-playground/   ← this repo
└── amlnn-toolkit/            ← SDK placed here automatically found
```

使用以下命令克隆：

```bash
git clone https://github.com/Amlogic-NN/amlnn-toolkit.git ../amlnn-toolkit
```

每个**示例**目录都包含 **build-android.sh** 和 **build-linux.sh** 脚本。编译步骤请参阅对应示例目录中 **README.zh-CN.md** 文件的**第 4 章**。

### Android 编译

Android 编译需要 NDK 工具链。构建脚本通过 `ANDROID_NDK_PATH` 环境变量查找 NDK 路径。

构建前请设置环境变量，例如：

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
```

> **注意：** 推荐使用 NDK **r25c**。下载地址：https://github.com/android/ndk/wiki/Unsupported-Downloads

如需**一次性构建所有示例**，请使用顶层批处理脚本：

```bash
cd examples
./build-android-all.sh          # auto-detects amlnn-toolkit
# or explicitly:
AMLNN_HOME=/path/to/amlnn-toolkit ./build-android-all.sh
```

该脚本会自动清理上一次的构建产物，按上述优先级规则解析 AMLNN SDK，并在结束时输出构建摘要。

### Yocto/Debian/Armbian 编译

每个示例的 `build-linux.sh` 也支持通过 `-m yocto` 参数启用 **Yocto** 模式。

**依赖：** 需要 Yocto SDK（Poky）。通过环境变量或 `-s` 参数设置路径：

```bash
export YOCTO_SDK_ROOT=/path/to/poky/sdk
```

工具链文件由所有示例共用，位于 `examples/cmake/yocto-toolchain.cmake`。

**构建单个示例：**

```bash
cd examples/yolox/cpp

# 64-bit (default)
./build-linux.sh -m yocto -s /path/to/poky/sdk

# 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/poky/32bit-sdk
```

**一次性构建所有示例：**

```bash
cd examples

# 64-bit
./build-linux-all.sh -m yocto -s /path/to/poky/sdk

# 32-bit
./build-linux-all.sh -m yocto -b 32 -s /path/to/poky/32bit-sdk

# Clean yocto build artifacts
./clean-linux-all.sh -m yocto
```

> **注意：** `LLMs` 示例会自动从批处理构建脚本中排除。

## 发布说明

| 版本  | 说明     |
| ----- | -------- |
| 1.0.0 | 首个版本 |

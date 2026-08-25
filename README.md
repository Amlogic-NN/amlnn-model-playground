

# Amlogic NN Model Playground

<div align="left">
  <img src="poster1.jpg" width="100%" alt="Amlogic Tech Banner">
</div>

## Introduction

**amlnn-model-playground** has been developed using the [amlnn-toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit). By completing the **model conversion** and **deployment** steps, we have created a ready-to-go **model zoo** for commonly used models. The demo package provides complete model conversion scripts, as well as a complete workflow for **Python API** and **C API** to run the converted model.

**Objective:** To help users get started and deploy models on the Amlogic NPU platform. The rich algorithm library in the model zoo can help guide developers through testing, benchmarking, proof of concept, and deployment of edge AI products.

## Dependencies

- **amlnn-runtime**: NPU runtime library, refer to [amlnn-runtime](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_runtime)
- **amlnn-toolkit**: model conversion tool, refer to [amlnn-toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit).
  - Installation package (`.whl`) paths:
    - [`amlnn_toolkit`](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit/whl): for `amlnn_toolkit` / `amlnn_edge_toolkit` wheels (model conversion & runtime)
    - [`amlnn_toolkit_lite`](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit_lite/amlnn_edge_toolkit_lite/whl/aarch64): for `amlnn_edge_toolkit_lite` wheels (model runtime only)
- **Cross-compilation toolchain**: For the complete toolchain installation guide, please refer to [Platform Compile Guide](docs/Platform_compile_guide.md).

## Support List

### CNN

| Category                  | Link                                               | Model Name                           | Quantized Dtype | Platform                          |
| ------------------------- | -------------------------------------------------- | ------------------------------------ | --------------- | --------------------------------- |
| Classification            | [mobilenet](examples/mobilenet/)                   | mobilenet_v2                         | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Classification            | [resnet](examples/resnet/)                         | resnet50-v2                          | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Classification            | [DINO](examples/DINO/)                             | DINO backbone + linear classifier    | w8a16           | A311Y3/C305X2/A123X               |
| Audio Classification      | [yamnet](examples/yamnet/)                         | YAMNet                               | w8a16           | A311Y3/C305X2/A123X               |
| Object Detection          | [DETR](examples/DETR/)                             | DETR                                 | w16a16          | A311Y3/C305X2/A123X               |
| Object Detection          | [ppyoloe](examples/ppyoloe/)                       | PP-YOLOE                             | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [qrcode](examples/qrcode/)                         | QR code detection                    | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yolov5](examples/yolov5/)                         | YOLOv5                               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yolov6](examples/yolov6/)                         | YOLOv6                               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yolov7](examples/yolov7/)                         | YOLOv7                               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yolov8](examples/yolov8/)                         | YOLOv8                               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yolov10](examples/yolov10/)                       | YOLOv10                              | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yolov11](examples/yolov11/)                       | YOLO11                               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yoloe](examples/yoloe/)                           | YOLOE                                | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yoloworld](examples/yoloworld/)                   | YOLO-World                           | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Object Detection          | [yolox](examples/yolox/)                           | YOLOX                                | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Face Detection            | [retinaface](examples/retinaface/)                 | RetinaFace                           | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Gesture Recognition       | [gesture](examples/gesture/)                       | Gesture                              | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Image Segmentation        | [deeplabv3](examples/deeplabv3/)                   | DeepLabV3                            | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Image Segmentation        | [mobilesam](examples/mobilesam/)                   | MobileSAM                            | w8a16           | A311Y3/C305X2/A123X               |
| Image Segmentation        | [ppseg](examples/ppseg/)                           | PP-LiteSeg                           | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Instance Segmentation     | [yolov5-seg](examples/yolov5-seg/)                 | YOLOv5 segmentation                  | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Instance Segmentation     | [yolov8-seg](examples/yolov8-seg/)                 | YOLOv8 segmentation                  | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Oriented Object Detection | [yolov8-obb](examples/yolov8-obb/)                 | YOLOv8 OBB                           | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Pose Estimation           | [blazepose_detect](examples/blazepose_detect/)     | BlazePose detection                  | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Pose Estimation           | [blazepose_landmark](examples/blazepose_landmark/) | BlazePose landmark                   | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Pose Estimation           | [yolov8-pose](examples/yolov8-pose/)               | YOLOv8 pose                          | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| License Plate Recognition | [lprnet](examples/lprnet/)                         | LPRNet                               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR Detection             | [ppocr-det](examples/ppocr-det/)                   | PaddleOCR detection                  | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR Detection             | [ppocr_det_v4](examples/ppocr_det_v4/)             | PaddleOCR detection v4               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR Detection             | [ppocrv5-det](examples/ppocrv5-det/)               | PaddleOCR detection v5               | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| OCR Recognition           | [ppocr_rec_v4](examples/ppocr_rec_v4/)             | PaddleOCR recognition v4             | w8a16           | A311Y3/C305X2/A123X               |
| OCR System                | [ppocr_v4_system](examples/ppocr_v4_system/)       | PaddleOCR v4 detection + recognition | w8a16           | A311Y3/C305X2/A123X               |
| OCR System                | [ppocrv5-system](examples/ppocrv5-system/)         | PP-OCRv5 system                      | w8a16           | A311Y3/C305X2/A123X               |
| OCR System                | [ppocrv6-system](examples/ppocrv6-system/)         | PP-OCRv6 system                      | w8a16           | A311Y3/C305X2/A123X               |
| Speech Recognition        | [sensevoice](examples/sensevoice/)                 | SenseVoice                           | w8a16           | A311Y3/C305X2/A123X               |
| Speech Recognition        | [whisper](examples/whisper/)                       | Whisper Tiny                         | w8a8            | A311D2/S905X5/A311Y3/C305X2/A123X |
| Image-Text Matching       | [clip](examples/clip/)                             | CLIP                                 | w8a16           | A311Y3/C305X2/A123X               |
| Image-Text Matching       | [mobileclip](examples/mobileclip/)                 | MobileCLIP                           | w8a16           | A311Y3/C305X2/A123X               |

### Large Language Models

| Model Name        | Quantized Dtype | Platform            |
| ----------------- | --------------- | ------------------- |
| DeepSeek-R1-1.5B  | w4a16           | A311Y3/C305X2/A123X |
| Qwen-1.8B         | w4a16           | A311Y3/C305X2/A123X |
| Qwen1.5-0.5B      | w4a16           | A311Y3/C305X2/A123X |
| Qwen1.5-1.8B      | w4a16           | A311Y3/C305X2/A123X |
| Qwen2-0.5B        | w4a16           | A311Y3/C305X2/A123X |
| Qwen2-1.5B        | w4a16           | A311Y3/C305X2/A123X |
| Qwen2.5-0.5B      | w4a16           | A311Y3/C305X2/A123X |
| Qwen2.5-1.5B      | w4a16           | A311Y3/C305X2/A123X |
| Qwen2.5-3B        | w4a16           | A311Y3/C305X2/A123X |
| Qwen3-0.6B        | w4a16           | A311Y3/C305X2/A123X |
| Qwen3-1.7B        | w4a16           | A311Y3/C305X2/A123X |
| Qwen3-4B          | w4a16           | A311Y3/C305X2/A123X |
| TinyLlama0.4-1.1B | w4a16           | A311Y3/C305X2/A123X |
| TinyLlama1-1.1B   | w4a16           | A311Y3/C305X2/A123X |
| Llama2-7B         | w4a16           | A311Y3/C305X2/A123X |
| Llama3.2-1B       | w4a16           | A311Y3/C305X2/A123X |
| Llama3.2-3B       | w4a16           | A311Y3/C305X2/A123X |
| OpenLLaMA-3B      | w4a16           | A311Y3/C305X2/A123X |
| Gemma1-2B         | w4a16           | A311Y3/C305X2/A123X |
| Gemma2-2B         | w4a16           | A311Y3/C305X2/A123X |
| Gemma3-270M       | w4a16           | A311Y3/C305X2/A123X |
| Gemma3-1B         | w4a16           | A311Y3/C305X2/A123X |
| Gemma4-E2B        | w4a16           | A311Y3/C305X2/A123X |
| Gemma4-E4B        | w4a16           | A311Y3/C305X2/A123X |
| Phi1.5-1.3B       | w4a16           | A311Y3/C305X2/A123X |
| Phi2-2.7B         | w4a16           | A311Y3/C305X2/A123X |
| Phi3-3.8B         | w4a16           | A311Y3/C305X2/A123X |
| ChatGLM3-6B       | w4a16           | A311Y3/C305X2/A123X |
| MiniCPM3-4B       | w4a16           | A311Y3/C305X2/A123X |
| MiniCPM4-0.5B     | w4a16           | A311Y3/C305X2/A123X |
| InternVL2-1B      | w4a16           | A311Y3/C305X2/A123X |
| InternVL3-1B      | w4a16           | A311Y3/C305X2/A123X |
| Qwen-VL2-2B       | w4a16           | A311Y3/C305X2/A123X |
| Qwen-VL2.5-3B     | w4a16           | A311Y3/C305X2/A123X |
| Qwen-VL3-2B       | w4a16           | A311Y3/C305X2/A123X |
| Qwen-VL3-4B       | w4a16           | A311Y3/C305X2/A123X |
| MobileVLM-V2-1.7B | w4a16           | A311Y3/C305X2/A123X |
| MiniCPM-V2.6-8B   | w4a16           | A311Y3/C305X2/A123X |

**Note:** w4a16 means that the weights are quantized using int4, and the features are using float16.

## Performance

Benchmark (FPS) and accuracy evaluation data has been moved to the dedicated [PERFORMANCE.md](PERFORMANCE.md) page.

## Examples Compile

### AMLNN Runtime setup

The C++ demos depend on the **AMLNN** runtime library. The build system automatically looks for `amlnn-toolkit` as a sibling directory:

```
modelzoo/
├── amlnn-model-playground/   ← this repo
└── amlnn-toolkit/            ← SDK placed here automatically found
```

Clone it with:

```bash
git clone https://github.com/Amlogic-NN/amlnn-toolkit.git ../amlnn-toolkit
```

Each **example** directory contains a **build-android.sh** and **build-linux.sh** script. For compilation steps, refer to **Chapter 4** of the **README.md** file in the corresponding example directory.

### Android Compilation

Android compilation requires the NDK toolchain. The build scripts look for the NDK path via the `ANDROID_NDK_PATH` environment variable.

Set environment variables before building, for example:

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
```

> **Note:** NDK **r25c** is recommended. Download: https://github.com/android/ndk/wiki/Unsupported-Downloads

To build **all examples at once**, use the top-level batch script:

```bash
cd examples
./build-android-all.sh          # auto-detects amlnn-toolkit
# or explicitly:
AMLNN_HOME=/path/to/amlnn-toolkit ./build-android-all.sh
```

The script automatically cleans the previous build, resolves the AMLNN SDK via the priority rules above, and prints a build summary at the end.

### Yocto/Debian/Armbian Compilation

Each example's `build-linux.sh` also supports **Yocto** mode via the `-m yocto` flag.

**Dependency:** A Yocto SDK (Poky). Set the path via environment variable or `-s` flag:

```bash
export YOCTO_SDK_ROOT=/path/to/poky/sdk
```

The toolchain file is shared across all demos at `examples/cmake/yocto-toolchain.cmake`.

**Build a single demo:**

```bash
cd examples/yolox/cpp

# 64-bit (default)
./build-linux.sh -m yocto -s /path/to/poky/sdk

# 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/poky/32bit-sdk
```

**Build all demos at once:**

```bash
cd examples

# 64-bit
./build-linux-all.sh -m yocto -s /path/to/poky/sdk

# 32-bit
./build-linux-all.sh -m yocto -b 32 -s /path/to/poky/32bit-sdk

# Clean yocto build artifacts
./clean-linux-all.sh -m yocto
```

> **Note:** The `LLMs` demo is automatically excluded from the batch build scripts.

## Release Notes

| Version | Description   |
| ------- | ------------- |
| 1.0.0   | First Version |

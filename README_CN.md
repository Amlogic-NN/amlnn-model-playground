[简体中文](README_CN.md) | [English](README.md)

![Banner](assets/poster.jpeg)

# amlnn-model-playground

# 简介

​		**amlnn-model-playground**是基于**amlnn toolkit**完成**模型转换**与**部署**，实现主流常用算法的model zoo。demo包中提供完整的模型转换脚本，以及使用 **Python API**，**OpenAI API** 和 **C API** 对转换后的模型进行推理运行的完整流程。

**目的：** 帮助用户快速上手，完成算法模型在 Amlogic NPU平台上的部署。model zoo中丰富的算法库可以更好的指导客户AI产品落地。



# 依赖项

- **amlnn-model-playground** 中的模型转换功能目前依赖于 Amlogic 提供的模型转换工具 **`adla-toolkit-binary-x.x.x.x`**，当前版本请联系您的销售代表获取该工具。
- **Android编译**依赖NDK工具链，当前建议使用**r25c**版本，下载链接:https://github.com/android/ndk/wiki/Unsupported-Downloads
- **Linux编译**依赖工具链:**gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf** ,下载链接：https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/

# 支持列表

| Category               | Model_name                                                   | Dtype  | Platform      |
| ---------------------- | ------------------------------------------------------------ | ------ | ------------- |
| Classification         | [mobilenet_v2](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) | INT8   | A311D2/S905X5 |
| Classification         | [resnet50-v2](https://github.com/onnx/models/blob/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/resnet/model/resnet50-v2-7.onnx) | INT8   | A311D2/S905X5 |
| Object Detection       | [yolov8](https://github.com/ultralytics/ultralytics)         | INT8   | A311D2/S905X5 |
| Object Detection       | [yolov11](https://github.com/ultralytics/ultralytics)        | INT8   | A311D2/S905X5 |
| Object Detection       | [yoloworld](https://github.com/AILab-CVC/YOLO-World)         | INT8   | A311D2/S905X5 |
| Object Detection       | [yoloe](https://github.com/ultralytics/ultralytics)          | INT8   | A311D2/S905X5 |
| Face Key Points        | [retinaface](https://github.com/biubug6/Pytorch_Retinaface)  | INT8   | A311D2/S905X5 |
| Text Detection         | ppocr-det                                                    | INT8   | A311D2/S905X5 |
| Pose Estimation        | blazepose_detect                                             | INT8   | A311D2/S905X5 |
| Pose Estimation        | blazepose_landmark                                           | INT8   | A311D2/S905X5 |
| Voiceprint recognition | [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)        | Hybrid | A311D2/S905X5 |
| Speech Recognition     | [whisper](https://github.com/openai/whisper)                 | Hybrid | A311D2/S905X5 |
| Image-Text Matching    | [clip](https://huggingface.co/openai/clip-vit-base-patch32)  | Hybrid | A311D2/S905X5 |
| Chat LLM               | deepseek                                                     | Hybrid | A311D2/S905X5 |



# Benchmark List(FPS)

| Examples           | Model_name   | input_shapes     | Dtype | S905X5 | A311D2 |
| ------------------ | ------------ | ---------------- | ----- | ------ | ------ |
| mobilenet          | mobilenet_v2 | [1, 3, 224, 224] | INT8  |1047.54 | 798.94 |
| resnet             | resnet50-v2  | [1, 3, 224, 224] | INT8  | 106.78 | 128.91 |
| yolov8             | yolov8l      | [1, 3, 640, 640] | INT8  | 11.55  | 11.12  |
| yolov11            | yolov11n     | [1, 3, 640, 640] | INT8  | 41.14  | 41.48  |
| yoloworld          | yoloworld    | [1, 3, 480, 640] | INT8  | 19.38  | 19.04  |
| yoloe              | yoloe        | [1, 3, 288, 512] | INT8 | 53.9 | 37.8 |
| retinaface         | retinaface   | [1, 3, 320, 320] | INT8  | 341.99 | 305.89 |
| ppocr-det          | paddleocrv4-det | [1, 3, 640, 640] | INT8 | 37.66 | 38.85 |
| blazepose_detect   | blazepose_detection | [1, 3, 224, 224] | INT8 | 476.29 | 461.74 |
| blazepose_landmark | blazepose_landmark_full | [1, 3, 256, 256] | INT16 | 84.59 | 70.31 |
| Whisper            | encoder_tiny_en |  [1, 80, 3000]     | Hybrid  |  0.71 |  0.58  |
| Whisper            | decoder_tiny_en |  [1, 1500, 384]&[1, 48]   | Hybrid  |  10.35  |  9.22  |
| Clip               | clip-vit-base-patch32 |  [1, 3, 224, 224]   | Hybrid  |  7.48  |  6.82  |

- 性能数据是使用natvie case测试出的模型在NPU上的运行时间，如无特殊说明，不包含前后处理的耗时。
- \表示暂时不支持。

# Examples 编译

​		每个**example**目录下面都有**build-android.sh** 和**build-linux.sh**脚本，编译步骤参考对应example目录下的**README.md**文件的**第四章节**



# **Release Notes**

| Version | Description   |
| ------- | ------------- |
| 1.0.0   | First Version |





​	
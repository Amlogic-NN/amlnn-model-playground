[简体中文](README_CN.md) | [English](README.md)
![Banner](poster.jpeg)

# amlnn-model-playground

# Introduction

​		**amlnn-model-playground** Has been developed using the **amlnn toolkit**, by completing the  **model conversion** and **deployment** steps, we have created a ready to go **model zoo** for commonly used models. The demo package provides complete model conversion scripts, as well as a complete workflow for **Python API**, **OpenAI API**, and **C API** to run the converted model.

**Objective:** To help users get started and deploy models on the Amlogic NPU platform. The rich algorithm library in the model zoo can help guide developers through test, benchmarking, proof of concept and deployment of edge AI products.

## Dependencies

- The model conversion functionality in **amlnn-model-playground** currently relies on the model conversion tool `adla-toolkit-binary-x.x.x.x` provided by Amlogic, please contact your sales representivie for access in the short term. 
- **Android compilation** depends on the NDK toolchain. Currently, version r25c is recommended. Download link: https://github.com/android/ndk/wiki/Unsupported-Downloads
- **Linux compilation** toolchain dependency: **gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf**, download link: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads/

# Support List

| Category               | Model_name                                                   | Dtype  | Platform      |
| ---------------------- | ------------------------------------------------------------ | ------ | ------------- |
| Classification         | [mobilenet_v2](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) | INT8   | A311D2/S905X5 |
| Classification         | [resnet50-v2](https://github.com/onnx/models/blob/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/resnet/model/resnet50-v2-7.onnx) | INT8   | A311D2/S905X5 |
| Object Detection       | [yolov8](https://github.com/ultralytics/ultralytics)         | INT8   | A311D2/S905X5 |
| Object Detection       | [yolov11](https://github.com/ultralytics/ultralytics)        | INT8   | A311D2/S905X5 |
| Object Detection       | [yoloworld](https://github.com/AILab-CVC/YOLO-World)         | INT8   | A311D2/S905X5 |
| Object Detection       | [yoloe](https://github.com/ultralytics/ultralytics)          | INT8   | A311D2/S905X5 |
| Object Detection       | [yolox](https://github.com/Megvii-BaseDetection/YOLOX/)      | INT8   | A311D2/S905X5 |
| Face Key Points        | [retinaface](https://github.com/biubug6/Pytorch_Retinaface)  | INT8   | A311D2/S905X5 |
| Text Detection         | ppocr-det                                                    | INT8   | A311D2/S905X5 |
| Pose Estimation        | blazepose_detect                                             | INT8   | A311D2/S905X5 |
| Pose Estimation        | blazepose_landmark                                           | INT8   | A311D2/S905X5 |
| Voiceprint recognition | [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)        | Hybrid | A311D2/S905X5 |
| Speech Recognition     | whisper                                                      | Hybrid | A311D2/S905X5 |
| Image-Text Matching    | [clip](https://huggingface.co/openai/clip-vit-base-patch32)  | Hybrid | A311D2/S905X5 |
| Chat LLM               | deepseek                                                     | Hybrid | A311D2/S905X5 |



# Benchmark List(FPS)

| Examples           | Model_name   | input_shapes     | Dtype | S905X5 | A311D2 |
| ------------------ | ------------ | ---------------- | ----- | ------ | ------ |
| mobilenet          | mobilenet_v2 | [1, 3, 224, 224] | INT8  |1047.54 | 798.94 |
| resnet             | resnet50-v2  | [1, 3, 224, 224] | INT8  | 106.78 | 128.91 |
| yolov8             | yolov8n      | [1, 3, 640, 640] | INT8  | 101.72 | 95.14  |
|                    | yolov8s      | [1, 3, 640, 640] | INT8  | 42.33  | 42.77  |
|                    | yolov8m      | [1, 3, 640, 640] | INT8  | 19.67  | 19.82  |
|                    | yolov8l      | [1, 3, 640, 640] | INT8  | 10.53  | 10.12  |
| yolov11            | yolov11n     | [1, 3, 640, 640] | INT8  | 41.14  | 41.48  |
| yoloworld          | yoloworld    | [1, 3, 480, 640] | INT8  | 19.38  | 19.04  |
| yoloe              | yoloe        | [1, 3, 288, 512] | INT8  | 53.9  | 37.8   |
| yolox              | yolox_tiny   | [1, 3, 640, 640] | INT8  | 42.9  | 35.89  |
|                    | yolox_s      | [1, 3, 640, 640] | INT8  | 35.12  | 33.31  |
|                    | yolox_m      | [1, 3, 640, 640] | INT8  | 18.55  | 17.98  |
| retinaface         | retinaface   | [1, 3, 320, 320] | INT8  | 341.99 | 305.89 |
| ppocr-det          | paddleocrv4-det | [1, 3, 640, 640] | INT8 | 37.66 | 38.85 |
| blazepose_detect   | blazepose_detection | [1, 3, 224, 224] | INT8 | 476.29 | 461.74 |
| blazepose_landmark | blazepose_landmark_full | [1, 3, 256, 256] | INT16 | 84.59 | 70.31 |
| Whisper            | encoder_tiny_en |  [1, 80, 3000]     | Hybrid  |  0.71 |  0.58  |
|                    | decoder_tiny_en |  [1, 1500, 384]&[1, 48]   | Hybrid  |  10.35  |  9.22  |
| Clip               | clip-vit-base-patch32 |  [1, 3, 224, 224]   | Hybrid  |  7.48  |  6.82  |

- The performance data represents the runtime of the model on the NPU, as tested using the native case. Unless otherwise specified, it does not include the time spent on pre- and post-processing.
- means currently supported.

# Examples Compile

​     Each **example** directory contains a **build-android.sh** and build-linux.sh **script**. For compilation steps, refer to **Chapter 4** of the **README.md** file in the corresponding example directory.



# **Release Notes**

| Version | Description   |
| ------- | ------------- |
| 1.0.0   | First Version |






​	



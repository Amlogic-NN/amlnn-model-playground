

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

## Benchmark List (FPS)

### CNN

| Example            | Model Name              | Input Shapes           | Dtype | S905X5  | A311D2 | A311Y3 |
| ------------------ | ----------------------- | ---------------------- | ----- | ------- | ------ | ------ |
| mobilenet          | mobilenet_v2            | [1, 3, 224, 224]       | w8a8  | 1047.54 | 798.94 | 1014.2 |
| resnet             | renet50_v2              | [1, 3, 224, 224]       | w8a8  | 127.49  | 136.65 | 266.88 |
| yolov8             | yolov8n                 | [1, 3, 640, 640]       | w8a8  | 101.72  | 95.14  | 191.06 |
|                    | yolov8s                 | [1, 3, 640, 640]       | w8a8  | 42.33   | 42.77  | 83.08  |
|                    | yolov8m                 | [1, 3, 640, 640]       | w8a8  | 19.67   | 19.82  | 35.30  |
|                    | yolov8l                 | [1, 3, 640, 640]       | w8a8  | 10.53   | 10.12  | 18.37  |
| yolov11            | yolov11n                | [1, 3, 640, 640]       | w8a8  | 41.14   | 41.48  | 62.24  |
| yoloworld          | yoloworld(v8m)          | [1, 3, 480, 640]       | w8a8  | 19.38   | 19.04  | 35.30  |
| yoloe              | yoloe                   | [1, 3, 288, 512]       | w8a8  | 53.9    | 37.8   | 59.03  |
| yolox              | yolox_tiny              | [1, 3, 640, 640]       | w8a8  | 42.9    | 35.89  | 56.51  |
|                    | yolox_s                 | [1, 3, 640, 640]       | w8a8  | 35.12   | 33.31  | 47.91  |
|                    | yolox_m                 | [1, 3, 640, 640]       | w8a8  | 18.55   | 17.98  | 27.54  |
| retinaface         | retinaface              | [1, 3, 320, 320]       | w8a8  | 341.99  | 305.89 | 269.18 |
| ppocr-det          | paddleocrv4-det         | [1, 3, 640, 640]       | w8a8  | 37.66   | 38.85  | 99.80  |
| blazepose_detect   | blazepose_detection     | [1, 3, 224, 224]       | w8a8  | 476.29  | 461.74 | 572.74 |
| blazepose_landmark | blazepose_landmark_full | [1, 3, 256, 256]       | w8a8  | 84.59   | 70.31  | 328.41 |
| whisper            | encoder_tiny_en         | [1, 80, 3000]          | w8a16(hybrid)  | 0.71    | 0.58   | 8.75   |
|                    | decoder_tiny_en         | [1, 1500, 384]&[1, 48] | w8a16(hybrid)  | 10.35   | 9.22   | 51.55  |
| clip               | clip-vit-base-patch32   | [1, 3, 224, 224]       | w8a16(hybrid)  | 7.48    | 6.82   | 73.00  |

### Large Language Models

| Model      | Version | Param | SoC        | Dtype  | SeqLen | Max Context | New Tokens | TTFT (ms) | Tokens/s | Memory (MB) |
| ---------- | ------- | ----- | ---------- | ------ | ------ | ----------- | ---------- | --------- | -------- | ----------- |
| DeepSeek   | R1      | 1.5B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 192.58    | 15.07    | 1157.12     |
| Qwen       | 1       | 1.8B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 233.59    | 10.53    | 1597.44     |
| Qwen       | 1.5     | 0.5B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 78.97     | 26.89    | 589.56      |
| Qwen       | 1.5     | 1.8B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 223.51    | 10.72    | 1546.24     |
| Qwen       | 2       | 0.5B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 73.82     | 36.17    | 349.03      |
| Qwen       | 2       | 1.5B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 197.25    | 14.49    | 1034.24     |
| Qwen       | 2.5     | 0.5B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 72.32     | 37.71    | 423.22      |
| Qwen       | 2.5     | 1.5B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 196.81    | 14.59    | 1034.24     |
| Qwen       | 2.5     | 3B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 359.76    | 8.02     | 1955.84     |
| Qwen       | 3       | 0.6B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 121.88    | 17.92    | 724.12      |
| Qwen       | 3       | 1.7B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 217.19    | 10.94    | 1484.8      |
| Qwen       | 3       | 4B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 486.45    | 5.12     | 2703.36     |
| TinyLlama  | 0.4     | 1.1B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 132.02    | 19.9     | 695.04      |
| TinyLlama  | 1       | 1.1B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 133.64    | 19.96    | 694.99      |
| Llama      | 2       | 7B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 910.59    | 2.63     | 4976.64     |
| Llama      | 3.2     | 1B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 144.23    | 16.62    | 840.93      |
| Llama      | 3.2     | 3B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 505.53    | 6.53     | 2181.12     |
| OpenLLaMA  | 1       | 3B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 568.08    | 3.78     | 2744.32     |
| Gemma      | 1       | 2B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 310.3     | 9.61     | 1761.28     |
| Gemma      | 2       | 2B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 377.7     | 7.59     | 2007.04     |
| Gemma      | 3       | 270M  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 97.76     | 44.49    | 583.17      |
| Gemma      | 3       | 1B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 200.1     | 18.61    | 1044.48     |
| Gemma      | 4       | E2B   | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 485.85    | 7.57     | 3848.13     |
| Phi        | 1.5     | 1.3B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 196.6     | 11.73    | 1259.52     |
| Phi        | 2       | 2.7B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 392.97    | 6.19     | 2334.72     |
| Phi        | 3       | 3.8B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 703.43    | 4.84     | 3194.88     |
| ChatGLM3   | 3       | 6B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 684.03    | 4.12     | 3840        |
| MiniCPM    | 3       | 4B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 597.18    | 2.77     | 3921.92     |
| MiniCPM    | 4       | 0.5B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 64.79     | 42.27    | 300.67      |
| InternVL   | 2       | 1B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 217.61    | 5.67     | 433.18      |
| InternVL   | 3       | 1B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 407.17    | 5.17     | 433.19      |
| Qwen-VL    | 2       | 2B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 186.63    | 15.06    | 1269.76     |
| Qwen-VL    | 2.5     | 3B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 372.4     | 7.55     | 2273.28     |
| Qwen-VL    | 3       | 2B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 486.09    | 8.58     | 1648.64     |
| Qwen-VL    | 3       | 4B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 1006.15   | 4.45     | 3123.2      |
| MobileVLM  | 2       | 1.7B  | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 319.15    | 6.09     | 1239.04     |
| MiniCPM-V  | 2.6     | 8B    | A9(A311Y3) | w4a16  | 128    | 2048        | 1920       | 1761.16   | 1.03     | 4669.44     |

- The performance data represents the runtime of the model on the NPU, as tested using the native case. Unless otherwise specified, it does not include the time spent on pre- and post-processing.

## Accuracy Evaluation

### CNN

| Model        | Task           | Input Shapes     | Dtype | Dataset / Validation      | Metric                                              | A311Y3 Latency (ms) | Memory (MB) | Bandwidth (MB) |
| ------------ | -------------- | ---------------- | ----- | ------------------------- | --------------------------------------------------- | ------------------- | ----------- | -------------- |
| MobileNetV2  | Classification | [1, 3, 224, 224] | w8a8  | ImageNet val1000          | Top1=67.00%<br>Top5=89.40%                          | 2.318               | 11.600      | 18.710         |
| MobileNetV2  | Classification | [1, 3, 224, 224] | w8a16 | ImageNet val1000          | Top1=69.20%<br>Top5=90.10%                          | 3.616               | 12.188      | 18.856         |
| ResNet50     | Classification | [1, 3, 224, 224] | w8a8  | ImageNet val1000          | Top1=72.40%<br>Top5=93.30%                          | 4.945               | 31.133      | 50.923         |
| ResNet50     | Classification | [1, 3, 224, 224] | w8a16 | ImageNet val1000          | Top1=75.90%<br>Top5=93.50%                          | 8.136               | 31.349      | 69.406         |
| YOLO11s      | Detection      | [1, 3, 640, 640] | w8a8  | COCO subset 300           | AP@0.5:0.95=47.83%<br>AP@0.5=65.11%<br>AP@0.75=51.06% | 10.507            | 16.917      | 80.378         |
| YOLOv5n      | Detection      | [1, 3, 640, 640] | w8a8  | COCO subset 300           | AP@0.5:0.95=31.96%<br>AP@0.5=49.60%<br>AP@0.75=34.34% | 4.206             | 6.768       | 22.556         |
| YOLOv5n      | Detection      | [1, 3, 640, 640] | w8a16 | COCO subset 300           | AP@0.5:0.95=32.65%<br>AP@0.5=49.45%<br>AP@0.75=35.41% | 19.248            | 73.518      | 178.187        |
| YOLOv8n      | Detection      | [1, 3, 640, 640] | w8a8  | COCO subset 300           | AP@0.5:0.95=40.70%<br>AP@0.5=56.95%<br>AP@0.75=43.38% | 17.904            | 73.066      | 160.516        |
| YOLOv8n      | Detection      | [1, 3, 640, 640] | w8a16 | COCO subset 300           | AP@0.5:0.95=41.66%<br>AP@0.5=57.67%<br>AP@0.75=45.19% | 28.697            | 73.863      | 202.247        |
| YOLOv8n-pose | Keypoints      | [1, 3, 640, 640] | w8a8  | COCO keypoints subset 300 | Box AP@0.5:0.95=45.16%<br>Keypoint AP@0.5:0.95=46.99% | 5.860            | 7.470       | 29.550         |
| YOLOv8n-pose | Keypoints      | [1, 3, 640, 640] | w8a16 | COCO keypoints subset 300 | Box AP@0.5:0.95=45.27%<br>Keypoint AP@0.5:0.95=48.14% | 19.320            | 74.708      | 209.853        |
| YOLOv8n-seg  | Box/Mask       | [1, 3, 640, 640] | w8a8  | COCO subset 300           | Box AP@0.5:0.95=34.93%<br>Mask AP@0.5:0.95=31.22%   | 6.712               | 9.404       | 48.942         |
| YOLOv8n-seg  | Box/Mask       | [1, 3, 640, 640] | w8a16 | COCO subset 300           | Box AP@0.5:0.95=40.80%<br>Mask AP@0.5:0.95=33.30%   | 32.171              | 36.570      | 160.160        |
| YOLOX-s      | Detection      | [1, 3, 640, 640] | w8a8  | COCO subset 300           | AP@0.5:0.95=43.28%<br>AP@0.5=63.00%<br>AP@0.75=46.80% | 11.199            | 15.581      | 54.532         |
| YOLOX-s      | Detection      | [1, 3, 640, 640] | w8a16 | COCO subset 300           | AP@0.5:0.95=44.67%<br>AP@0.5=63.93%<br>AP@0.75=48.42% | 29.506            | 81.984      | 231.351        |

### Large Language Models

#### PPL

| Model                         | PPL (GGUF FP16) | PPL (ADLA W4A16) | Change |
| ----------------------------- | --------------- | --------------- | ------ |
| Phi-3.5-mini-instruct         | 6.42            | 7.98            | +1.56  |
| Llama-2-7B-chat               | 7.63            | 8.86            | +1.23  |
| open_llama_3b                 | 7.79            | 9.41            | +1.62  |
| Llama-3.1-8B-Hermes-3         | 8.10            | 8.31            | +0.21  |
| TinyLlama-1.1B-Chat-v1.0      | 8.45            | 9.73            | +1.28  |
| Qwen2.5-1.5B-Instruct         | 10.15           | 12.11           | +1.96  |
| phi-2                         | 10.29           | 10.43           | +0.14  |
| Qwen2-1.5B-Instruct           | 10.43           | 11.23           | +0.80  |
| Llama-3.2-3B-Instruct         | 10.53           | 14.55           | +4.02  |
| Qwen3-4B-Instruct-2507        | 10.83           | 11.38           | +0.55  |
| TinyLlama-1.1B-Chat-v0.4      | 11.33           | 11.94           | +0.61  |
| gemma-2-2b-it                 | 12.88           | 11.82           | -1.06  |
| Llama-3.2-1B-Instruct         | 14.00           | 14.79           | +0.79  |
| Qwen1.5-1.8B-Chat             | 14.84           | 16.50           | +1.66  |
| Qwen2-0.5B-Instruct           | 15.12           | 17.27           | +2.15  |
| Qwen2.5-0.5B-Instruct         | 15.19           | 16.94           | +1.75  |
| MiniCPM3-4B                   | 16.26           | 10.00           | -6.26  |
| Qwen3-1.7B                    | 17.15           | 20.59           | +3.44  |
| gemma-3-4b-it                 | 17.25           | 33.47           | +16.22 |
| Qwen-1.8B-chat                | 21.21           | 22.58           | +1.37  |
| Qwen3-0.6B                    | 21.90           | 23.71           | +1.81  |
| gemma-2b-it                   | 22.99           | 22.59           | -0.40  |
| phi-1_5                       | 23.16           | 21.79           | -1.37  |
| gemma-3-1b-it                 | 29.12           | 35.37           | +6.25  |
| Qwen1.5-0.5B-Chat             | 31.39           | 28.43           | -2.96  |
| ChatGLM3-6B                   | 34.98           | 48.79           | +13.81 |
| DeepSeek-R1-Distill-Qwen-1.5B | 42.98           | 71.19           | +28.21 |
| gemma-3-270m-it               | 58.94           | 68.42           | +9.48  |

#### MMLU

| Model                                 | Source       | GGUF FP16 (PC GPU) | ADLA W4A16 (A311Y3) | Accuracy Change |
| ------------------------------------- | ------------ | ------------------ | ----------------------- | --------------- |
| MiniCPM3-4B                           | Hugging Face | 0.6400             | 0.6160                  | -4%             |
| Phi-3-mini-4k-instruct                | Hugging Face | 0.5660             | 0.5740                  | +1%             |
| Qwen2.5-0.5B-Instruct                 | Hugging Face | 0.4360             | 0.4300                  | -1%             |
| Qwen3-4B-Instruct-2507 (Non-thinking) | Hugging Face | 0.7520             | 0.7300                  | -3%             |
| Qwen3-4B (Thinking)                   | Hugging Face | 0.7803             | 0.8140                  | +4%             |

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

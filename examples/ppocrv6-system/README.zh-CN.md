[EN](README.md) | **CN**

# PP-OCRv6 系统

本示例使用 AMLNN 运行 [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) 文本检测与文本识别模型。

本目录中的示例代码演示以下内容：

1. 准备 PP-OCRv6 ONNX 模型
2. 将 ONNX 模型转换为 ADLA
3. 使用 ADLA 模型运行 C++ 示例
4. 查看推理结果

## 目录结构

```text
examples/ppocrv6-system/
|-- cpp/                  # C++ demo and build scripts
|-- model/                # Put ONNX and ADLA models here
|-- py/                   # ADLA export scripts
|-- ppocrv6_dict.txt      # Recognition dictionary file
```

## 许可证

本示例代码采用 Apache License 2.0 许可证。详情请参见仓库根目录的 [LICENSE](../../LICENSE) 文件。

## 1. 准备 PP-OCRv6 ONNX 模型

PP-OCRv6 模型可通过 Paddle2ONNX 工具从 PaddleOCR v6 导出，也可从官方 PP-OCRv6 模型发布页下载。

PP-OCRv6 检测模型期望的输入名为 `x`，形状为：

- `float32[DynamicDimension.0, 3, DynamicDimension.1, DynamicDimension.2]`

PP-OCRv6 识别模型期望的输入名为 `x`，形状为：

- `float32[DynamicDimension.0, 3, 48, DynamicDimension.1]`

两个模型均只暴露一个名为 `fetch_name_0` 的输出张量。

## 2. 将 ONNX 模型转换为 ADLA

### 导出 PP-OCRv6 模型

使用提供的导出脚本将检测和识别 ONNX 模型均转换为 ADLA。

```bash
python3 py/export_adla.py \
    --det-onnx ./model/det_ppocrv6.onnx \
    --rec-onnx ./model/rec_ppocrv6.onnx \
    --det-dataset-path ./det_dataset.txt \
    --rec-dataset-path ./rec_dataset.txt \
    --target-platform 007 \
    --out-dir ./model
```

该导出脚本已按 PP-OCRv6 的输入/输出节点名称配置，并采用适合 v6 推理的 ADLA 量化设置。

## 3. 运行 C++ 示例

### Android 构建

**前置条件：**

- Android NDK（建议 r25e）
- 已设置 `ANDROID_NDK_PATH` 环境变量

**构建：**

```bash
cd examples/ppocrv6-system/cpp
./build-android.sh -a arm64-v8a
```

**运行：**

```bash
adb push build/android/paddleocr_sys_demo /data/local/tmp/
adb push model/det_ppocrv6.adla model/rec_ppocrv6.adla /data/local/tmp/
adb push ppocrv6_dict.txt /data/local/tmp/
adb push test.jpg /data/local/tmp/

adb shell
cd /data/local/tmp
chmod +x paddleocr_sys_demo
export LD_LIBRARY_PATH=/vendor/lib64
./paddleocr_sys_demo \
    --image_path=/data/local/tmp/test.jpg \
    --det_model_path=/data/local/tmp/det_ppocrv6.adla \
    --rec_model_path=/data/local/tmp/rec_ppocrv6.adla \
    --dict_path=/data/local/tmp/ppocrv6_dict.txt
```

### Linux 构建

如果更倾向于 Linux，可使用 `examples/ppocrv6-system/cpp/src` 下的同一 CMake 工程，并在 Linux OpenCV 环境中构建。

## 4. 结果

示例会将可视化后的 OCR 结果以 `ocr_result_<input_image_filename>` 为文件名保存到当前工作目录。

如果在设备上运行，可用以下命令取出结果：

```bash
adb pull /data/local/tmp/ocr_result_test.jpg
```

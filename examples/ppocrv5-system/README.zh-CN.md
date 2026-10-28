[EN](README.md) | **CN**

# PP-OCRv5 系统示例

本示例使用 AMLNN 运行 [PP-OCRv5](https://github.com/PADDLEPADDLE/PADDLEOCR) 的 OCR 检测与 OCR 识别模型。完整流程如下：

1. 准备 ONNX 模型
2. 将 ONNX 模型转换为 ADLA 模型
3. 使用 ADLA 模型运行 C++ 示例
4. 查看结果图像

## 目录结构

```text
examples/ppocrv5-system/
|-- cpp/                  # C++ demo and build scripts
|-- model/                # Put ONNX and ADLA models here
|-- ocr_result_test.jpg   # Example detection output
```

## 许可证

本示例代码基于 Apache License 2.0 许可证发布。详情请参见仓库根目录的 [LICENSE](../../LICENSE) 文件。

PP-OCRv5 模型由第三方提供。重新分发或商业使用前，请确认并遵守模型来源方的许可证与使用条款。

本示例中的模型预处理和后处理代码由 AI 生成。若其中某些部分与已有工作相似而引起疑虑，请联系我们，我们将删除或调整相关代码。

## 1. 准备 ONNX 模型

### 导出 ONNX

模型来源：https://www.paddleocr.ai/main/version3.x/module_usage/module_overview.html

使用 PaddleX 的 Paddle2ONNX 插件将 Paddle 模块转换为 ONNX：https://www.paddleocr.ai/main/version3.x/inference_deployment/others/obtaining_onnx_models.html

## 2. 将 ONNX 转换为 ADLA

### det 模型

```bash
# Download quantitative image sets from the web and generate the dataset.txt

# convert
python3 export_adla.py \
        --det-onnx ./det_mobile_sim_static.onnx \
        --rec-onnx ./rec_mobile_sim_static.onnx \
        --det-dataset-path ./det_dataset.txt \
        --rec-dataset-path ./rec_dataset.txt \
        --target-platform 007
```

## 3. 运行 Python 示例

## 4. 运行 C++ 示例

### Android 构建

**前置条件：**

- Android NDK（推荐 r25e）
- 已设置 `ANDROID_NDK_PATH` 环境变量

**构建：**

可执行文件将生成在 `build/android/paddleocr_sys_demo`（注意：可执行文件名可能不同，请在构建目录中确认）。


**运行：**

```bash
# Push executable to device
adb push build/android/paddleocr_sys_demo /data/local/tmp/
adb push model/det_mobile_sim_static_w8a16.adla model/rec_mobile_sim_static_w8a16.adla /data/local/tmp/
adb push test.jpg /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x /data/local/tmp/paddleocr_sys_demo
export LD_LIBRARY_PATH=/vendor/lib64
./paddleocr_sys_demo \
    --image_path=/data/local/tmp/test.jpg \
    --det_model_path=/data/local/tmp/det_mobile_sim_static_w8a16.adla \
    --rec_model_path=/data/local/tmp/rec_mobile_sim_static_w8a16.adla \
    --dict_path=/data/local/tmp/ppocrv5_dict.txt
```

## 5. 结果

示例会在当前目录输出一张图像，其中包含检测到的文本框和识别出的文字，文件名为 ocr_result_<input_image_filename>。

从设备拉取结果图像：

```bash
adb pull /data/local/tmp/ocr_result_test.jpg
```

结果示例：

![ppocrv5-system 结果](./ocr_result_test.jpg)

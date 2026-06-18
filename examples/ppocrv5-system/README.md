# PaddleOCR Detection

## 1.Overview

[PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR)

## 2.Model Download

## 3. Model Conversion

#### det_mobile_sim_static.onnx
```bash
../../adla-toolkit-binary-3.2.9.4/bin/adla_convert \
        --model-type onnx \
        --model ./det_mobile_sim_static.onnx \
        --inputs "x" \
        --input-shapes "3,960,960" \
        --quantize-dtype int8 \
        --target-platform PRODUCT_PID0XA003 \
        --source-file ./ocr_npy/dataset.txt \
        --disable-per-channel false
```

#### rec_mobile_sim_static.onnx

```bash
../../adla-toolkit-binary-3.2.9.4/bin/adla_convert \
        --model-type onnx \
        --model ./rec_mobile_sim_static.onnx \
        --inputs "x" \
        --input-shapes "3,48,320" \
        --quantize-dtype int8 \
        --channel-mean-value "127.5,127.5,127.5,127.5" \
        --target-platform PRODUCT_PID0XA003 \
        --source-file ./ocr_pic/dataset.txt \
        --disable-per-channel false
```

## 4. Demo Run

### CPP

#### 1. Compile

**Prerequisites:**

- Android NDK (r25e recommended)
- `ANDROID_NDK_PATH` environment variable set

**Build:**

The executable will be generated at `build/android/paddleocr_sys_demo` (Note: executable name may vary, verify in build folder).

#### 2. Run

```bash
# Push executable to device
# Build for arm64-v8a
adb push build/android/paddleocr_sys_demo /data/local/tmp/
adb push model/det_mobile_sim_static_int8.adla rec_mobile_sim_static_int8.adla /data/local/tmp/
adb push test.jpg /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x /data/local/tmp/paddleocr_sys_demo
export LD_LIBRARY_PATH=/vendor/lib64
./paddleocr_sys_demo \
    --image_path=/data/local/tmp/test.jpg \
    --det_model_path=/data/local/tmp/det_mobile_sim_static_int8.adla \
    --rec_model_path=/data/local/tmp/rec_mobile_sim_static_int8.adla \
    --dict_path=/data/local/tmp/ppocrv5_dict.txt
```

### Python


# blazepose_detect

## 1.Overview

​		BlazePose Detection was introduced by Google as part of the MediaPipe framework, providing fast and lightweight person detection optimized for real-time performance on mobile and edge devices. The detector identifies the human region of interest (ROI) in an image, ensuring stable and efficient pose tracking in subsequent stages.

## 2.Model Download

- **Open Source model**

  - **Open Source projects:** https://github.com/google-ai-edge/mediapipe/tree/master

  - **Download weights**

    wget https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite

## 3. Model Conversion

```
cd model
Usage:   ./adla_convert.sh model_path adla_toolkit_path target_platform

example
 ./adla_convert.sh pose_detection.tflite /xxxx/adla-toolkit-binary-3.2.9.3 PRODUCT_PID0XA005
 ./adla_convert.sh pose_detection.tflite /xxxx/adla-toolkit-binary-3.2.9.3 PRODUCT_PID0XA005
 ./adla_convert.sh pose_detection.tflite /xxxx/adla-toolkit-binary-3.2.9.3 PRODUCT_PID0XA005
```

| Parameter         | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| model_path        | onnx model path                                              |
| adla_toolkit_path | path to adla_toolkit                                         |
| target_platform   | Specify target platform. for A311D2: PRODUCT_PID0XA003. for S905X5: PRODUCT_PID0XA005 |



## 4. Demo Run

### CPP

#### 1. Compile

**Prerequisites:**

- Android NDK (r25e recommended)
- `ANDROID_NDK_PATH` environment variable set

**Build:**
```bash
# Build for arm64-v8a
cd examples/blazepose_detect/cpp
./build-android.sh -a arm64-v8a
```

The executable will be generated at `build/android/blazepose_detect_demo` (Note: executable name may vary, verify in build folder).

#### 2. Run

```bash
# Push executable to device
adb push build/android/blazepose_detect_demo /data/local/tmp/
adb push model/blazepose_detect_int8_A311D2.adla /data/local/tmp/
adb push test_image.jpg /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x blazepose_detect_demo
export LD_LIBRARY_PATH=/vendor/lib64 or (/vendor/lib)

# Usage: ./blazepose_detect_demo <model_path> <image_path>
./blazepose_detect_demo blazepose_detect_int8_A311D2.adla test_image.jpg"
```

**Note:** Replace `blazepose_detect_int8_A311D2.adla` with your actual model file path.

### Python

**Prerequisites:**
- Python 3.10
- Required packages: `numpy`, `opencv-python`, `amlnnlite`

**Install dependencies:**
```bash
pip install numpy opencv-python amlnnlite-1.0.0-cp310-cp310-linux_aarch64.whl
```

**Run on device:**
```bash
python blazepose_detect.py --model-path ./blazepose_detect_int8_A311D2.adla
```

The script will automatically process all image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) in the current directory and save results to a `{model_name}_result` folder.

## 5.Results
The program will print the detection count and inference time. The result image with bounding boxes will be saved to the specified output path (`result.jpg` by default).


You can pull the result image back to view it:
```bash
adb pull result.jpg.
```
![alt text](result.jpg)


## Demo Run

### CPP

#### 1. Compile

**Prerequisites:**
- Android NDK (r25e recommended)
- `ANDROID_NDK_PATH` environment variable set

**Build:**
```bash
# Build for arm64-v8a
cd examples/clip/cpp
./build-android.sh -a arm64-v8a
```

The executable will be generated at `build/android_arm64-v8a/clip_demo` (Note: executable name may vary, verify in build folder).

#### 2. Run

```bash
# Push executable to device
adb push build/android_arm64-v8a/clip_demo /data/local/tmp/
adb push model/vision_model_int8_A311D2.adla /data/local/tmp/
adb push clip_datasets/ /data/local/tmp/
adb push test_hat_0.jpg /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x clip_demo
export LD_LIBRARY_PATH=/vendor/lib64 or (/vendor/lib)

# Usage: ./clip_demo <model_path> [base_dir] [json_filename]
./clip_demo vision_model_int8_A311D2.adla ./clip_datasets/ clip_text_res.json
```

**Note:** 
- Replace `vision_model_int8_A311D2.adla` with your actual model file path.
- The `base_dir` and `json_filename` parameters are optional. You can also use environment variables `CLIP_BASE_DIR` and `CLIP_JSON_FILENAME`.
- The program will prompt you to enter image paths interactively. Enter "exit" to quit.

### Python

**Prerequisites:**
- Python 3.10
- Required packages: `numpy`, `Pillow`, `amlnnlite`

**Install dependencies:**
```bash
pip install numpy Pillow amlnnlite-1.0.0-cp310-cp310-linux_aarch64.whl
```

**Run on device:**
```bash
# Basic usage (process current directory)
python clip.py --model-path ./vision_model_int8_A311D2.adla

# Specify image directory or file
python clip.py --model-path ./vision_model_int8_A311D2.adla --image-dir ./

# Specify base directory and JSON filename
python clip.py --model-path ./vision_model_int8_A311D2.adla --base-dir ./clip_datasets/ --json-filename clip_text_res.json
```

The script will automatically process all image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) in the specified directory or process a single image file, and display the best matching dataset for each image.

5. Results

The program will print the best matching dataset path for each processed image. The program searches through all dataset folders in the base directory and finds the text feature with the highest similarity to the input image.

**Example output:**
```
# python demo result
Model initialized successfully.

Found 2 image file(s) to process
Searching in base directory: ./clip_datasets/

Processing image: test_jacket_0.jpg
  Best matching dataset: ./clip_datasets/shirt10_jacket7
Searching in base directory: ./clip_datasets/

Processing image: test_hat_0.jpg
  Best matching dataset: ./clip_datasets/hat1_jd

Total results: 2
Index[0]: ./clip_datasets/shirt10_jacket7
Index[1]: ./clip_datasets/hat1_jd

Done.
```

The program returns the dataset folder path that contains the text feature with the highest similarity to the input image. Each result represents the best matching dataset for the corresponding input image.

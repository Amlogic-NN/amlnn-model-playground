
##  Demo Run

### CPP

#### 1. Compile

**Prerequisites:**
- Android NDK (r25e recommended)
- `ANDROID_NDK_PATH` environment variable set

**Build:**
```bash
# Build for arm64-v8a
cd examples/mobilenet/cpp
./build-android.sh -a arm64-v8a
```

The executable will be generated at `build/android/mobilenet_v2_demo` (Note: executable name may vary, verify in build folder).

#### 2. Run

```bash
# Push executable to device
adb push build/android/mobilenet_v2_demo /data/local/tmp/
adb push model/mobilenet_v2_1.0_224_quant_A311D2.adla /data/local/tmp/
adb push model/cat_224x224.jpg /data/local/tmp/
adb push model/labels.txt /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x mobilenet_v2_demo
export LD_LIBRARY_PATH=/vendor/lib64 or (/vendor/lib)

# Usage: ./mobilenet_v2_demo <model_path> <image_path> <labels_path>
./mobilenet_v2_demo mobilenet_v2_1.0_224_quant_A311D2.adla cat_224x224.jpg labels.txt
```

**Note:** Replace `mobilenet_v2_1.0_224_quant_A311D2.adla` with your actual model file path.

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
# Basic usage
python mobilenetv2.py --model-path ./mobilenet_v2_1.0_224_quant_A311D2.adla

# Run with performance testing (100 cycles)
python mobilenetv2.py --model-path ./mobilenet_v2_1.0_224_quant_A311D2.adla --run-cycles 100
```

The script will automatically process all image files (`.jpg`, `.jpeg`, `.png`, `.bmp`) in the current directory and display top-5 classification results for each image.

## Results

The program will print the top-5 classification results with probabilities for each processed image. 

**Example output:**
```
# python demo result
============================================================
Processing image 1/3: dog_224x224.jpg
============================================================

Top-5 Classification Results:
  1. Shih-Tzu                  (probability: 0.9239)
  2. Pekinese                  (probability: 0.0476)
  3. Lhasa                     (probability: 0.0263)
  4. Brabancon griffon         (probability: 0.0004)
  5. Dandie Dinmont            (probability: 0.0003)

============================================================
Processing image 2/3: cat_224x224.jpg
============================================================

Top-5 Classification Results:
  1. tiger cat                 (probability: 0.4774)
  2. tabby                     (probability: 0.4324)
  3. Egyptian cat              (probability: 0.0542)
  4. lynx                      (probability: 0.0150)
  5. Persian cat               (probability: 0.0025)

============================================================
Processing image 3/3: fish_224x224.jpeg
============================================================

Top-5 Classification Results:
  1. goldfish                  (probability: 0.9998)
  2. conch                     (probability: 0.0001)
  3. trifle                    (probability: 0.0000)
  4. axolotl                   (probability: 0.0000)
  5. American lobster          (probability: 0.0000)
```

The classification results show the model's confidence scores (probabilities) for each detected class, with the highest probability indicating the most likely classification.


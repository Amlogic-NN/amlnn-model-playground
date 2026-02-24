# CLIP

## 1. Overview

This demo demonstrates how to run CLIP (Contrastive Language-Image Pre-Training) image-text matching using AMLNNLite. The CLIP model consists of two parts: a vision encoder and a text encoder, which work together to compute similarity between images and text descriptions.

## 2. Model Download

TO DO

## 3. Model Conversion

TO DO

## 4. Demo Run

### CPP

#### 1. Compile

#### AMLNN SDK Setup

Resolve the AMLNN nnsdk dependency using one of the following methods:

- **Priority 1 – Environment variable (recommended)**
  ```bash
  export AMLNN_HOME=/path/to/amlnn-toolkit
  ```
- **Priority 3 – Sibling directory fallback** *(automatic)*
  Place `amlnn-toolkit` as a sibling to `amlnn-model-playground`:
  ```bash
  git clone git@github.com:Amlogic-NN/amlnn-toolkit.git ../amlnn-toolkit
  ```

**Prerequisites:**
- Android NDK (r25e recommended)
- `ANDROID_NDK_PATH` environment variable set

**Build:**
```bash
# Build for arm64-v8a
cd examples/clip/cpp
./build-android.sh -a arm64-v8a
```

The executable will be generated at `build/android_arm64-v8a/clip_demo`.

#### 2. Run

```bash
# Push executable and resources to device
adb push build/android_arm64-v8a/clip_demo /data/local/tmp/
adb push model/vision_model_int8_S905X5.adla /data/local/tmp/
adb push model/text_model_int8_S905X5.adla /data/local/tmp/
adb push tokenizer_path/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x clip_demo
export LD_LIBRARY_PATH=/vendor/lib64 or (/vendor/lib)

# Usage: ./clip_demo <vision_model> <text_model> <tokenizer_path> [--profiling]
./clip_demo vision_model_int8_S905X5.adla text_model_int8_S905X5.adla ./tokenizer_path/
```

The program will prompt for image paths and text descriptions interactively. Enter the path to an image file, then enter comma-separated text descriptions (or `skip` to use defaults). Type `exit` to quit.

**Argument Descriptions:**

| Argument       | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| vision_model   | Path to vision encoder .adla model (required)                |
| text_model     | Path to text encoder .adla model (required)                  |
| tokenizer_path  | Path to directory containing `vocab.json` and `merges.txt` (required) |
| --profiling    | Enable performance profiling output (optional)               |

**Note:** The `tokenizer_path` should contain `vocab.json` and `merges.txt` files from the CLIP tokenizer (e.g., from `openai/clip-vit-base-patch32`).

### Python

**Prerequisites:**
- Python 3.10
- Required packages: `numpy`, `Pillow`, `transformers`, `amlnnlite`

**Install dependencies:**
```bash
pip install numpy Pillow transformers amlnnlite-1.0.0-cp310-cp310-linux_aarch64.whl
```

**Run on device:**
```bash
python clip.py \
    --vision-model ./vision_model_int8_S905X5.adla \
    --text-model ./text_model_int8_S905X5.adla \
    --tokenizer-dir ./tokenizer_path \
    --image-path ./000000004505.jpg \
    --texts "a red handbag" "a blue jacket" "a red bus"
```

**Interactive Mode (Recommended):**

If you don't provide `--image-path`, the program will run in interactive mode:

```bash
python clip.py \
    --vision-model ./vision_model_int8_S905X5.adla \
    --text-model ./text_model_int8_S905X5.adla \
    --tokenizer-dir ./tokenizer_path
```

The program will prompt for image paths and text descriptions. Enter an image path to process, then enter comma-separated texts to compare. Type `exit` to quit.

**Argument Descriptions:**

| Argument         | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| --vision-model   | Path to vision encoder .adla model (required)                |
| --text-model     | Path to text encoder .adla model (required)                  |
| --tokenizer-dir  | Path to CLIPTokenizer directory (required)                   |
| --image-path     | Path to input image (.jpg, .png) - optional, will prompt if not provided |
| --texts          | List of text descriptions to compare (space-separated)       |
| --max-len        | Maximum token sequence length, default is 64                 |
| --logit-scale    | Logit scale factor, default is 100.0                         |

**Note:** The `--tokenizer-dir` should point to the directory containing the CLIPTokenizer files. You can use a Hugging Face model ID (e.g., `openai/clip-vit-base-patch32`) or a local directory.

## 5. Results

**Performance Feedback**

By using the `--profiling` flag (C++) or setting the loglevel to INFO, the program provides real-time performance metrics upon completion. The console log will display essential hardware and execution details, including:
- Hardware Information: System and ADLA library versions.
- Model Overview: Basic input/output configurations.
- NPU Metrics: Total inference time (latency) and total DRAM bandwidth consumption.

**Interactive Mode Example:**

```bash
$ ./clip_demo vision_model_int8_S905X5.adla text_model_int8_S905X5.adla ./tokenizer_path

[Info] Models initialized successfully.

============================================================
[Info] Image Path (or 'exit' to quit):
000000004505.jpg
[Info] Enter text descriptions (comma-separated, or 'skip' for defaults):
a red handbag, a blue jacket, a red bus

[Info] Processing image: 000000004505.jpg
[Info] Image embedding size: 512
[Info] Processing 3 text(s)...
[Info] Text embeddings size: 3 x 512

============================================================
CLIP Image-Text Matching Results
============================================================
Image: 000000004505.jpg
logit_scale: 100.000000
------------------------------------------------------------
[1] prob=0.999975  sim=0.327895  text='a red bus'
[2] prob=0.000016  sim=0.217690  text='a red handbag'
[3] prob=0.000008  sim=0.211029  text='a blue jacket'
============================================================

============================================================
[Info] Image Path (or 'exit' to quit):
exit
[Info] Exiting...
Free vision model memory.
Free text model memory.
[Info] Done.
```

# SenseVoice

## 1.Overview

SenseVoice is a multilingual speech foundation model developed by [FunAudioLLM](https://github.com/FunAudioLLM/SenseVoice). This demo deploys **SenseVoice-Small** on Amlogic ADLA NPU platforms for **offline** speech recognition.

**Capabilities**


| Capability | Description                                             |
| ---------- | ------------------------------------------------------- |
| ASR        | Multilingual speech-to-text                             |
| LID        | Spoken language identification                          |
| SER        | Speech emotion recognition                              |
| AED        | Acoustic event detection (e.g. speech, laughter, cough) |
| ITN        | Optional inverse text normalization (`--itn 0/1`)       |


**Supported languages:** `auto`, `zh`, `en`, `ja`, `ko`, `yue`

**Input audio:** PCM16 WAV, mono, **16 kHz** recommended.

**Limitations:** One-shot recognition only; max audio length is about **6 seconds** (~100 frames after LFR).

**Inference pipeline**

```text
WAV (16 kHz PCM16)
  → Fbank + LFR feature extraction (CPU, kaldi-native-fbank)
  → ADLA model inference (3 inputs: text_norm, language, features)
  → CTC greedy decoding
  → language / emotion / event / ITN tags + text
```

**Directory layout**

```text
examples/sensevoice/
├── model/
│   └── export-onnx-3input.py    # ONNX export script (run in SenseVoice repo, see Section 2)
├── input/                       # test wav files (en/zh/ja/ko/yue)
├── cpp/                         # C++ demo (build-android.sh / build-linux.sh)
└── py/                          # Python demo (sensevoice.py)
```

**Supported platforms:** A311D2, S905X5 (and other ADLA platforms with matching `.adla` model)

---

## 2. Export ONNX

This demo uses a **3-input ONNX** variant (`x`, `language`, `text_norm`) tailored for ADLA deployment. The export script is:

`model/export-onnx-3input.py`

### Prerequisites

Run the export inside the **[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)** repository (which provides `model.py`, `utils/`, etc.). The script in this demo directory is a copy for reference.

```bash
# 1. Clone SenseVoice
git clone https://github.com/FunAudioLLM/SenseVoice.git
cd SenseVoice

# 2. Install dependencies
pip install -r requirements.txt
pip install onnx onnxruntime

# 3. Copy export script from this demo
cp /path/to/amlnn-model-playground/examples/sensevoice/model/export-onnx-3input.py .
```

On first run, the script downloads `iic/SenseVoiceSmall` weights from ModelScope (network required).

### Export steps

```bash
cd SenseVoice
python export-onnx-3input.py
```

**Output directory:** `onnx-3input/` (created next to the script)


| Output file                              | Description                                                                |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| `onnx-3input/sensevoice_small.onnx`      | FP32 ONNX (opset 13, dynamic time axis) — **use this for ADLA conversion** |
| `onnx-3input/sensevoice_small.int8.onnx` | Dynamic-quantized ONNX (MatMul, QUInt8)                                    |
| `onnx-3input/tokens.txt`                 | Token table for CTC decoding                                               |


Keep `onnx-3input/sensevoice_small.onnx` and `onnx-3input/tokens.txt` for the next conversion step and demo run.

---

## 3. Model Conversion

Convert the ONNX model to ADLA format using `adla-toolkit-binary` from Amlogic.

> **Note:** Model conversion relies on `adla-toolkit-binary` provided by Amlogic. Please contact your sales representative for access.

### S905X5 example

Take **S905X5** as an example. Enter the `adla-toolkit-binary` directory and run:

```bash
# Export intermediate tflite / enable hybrid quantize
export ADLA_EXPORT_MIDDLE_TO_TFLITE=True
export ADLA_ENABLE_LLM_HYBRID_QUANTIZE=True
export ADLA_SET_EXTREME_VALUE=16000

# Random quantization dataset
export ADLA_SET_RANDOM_MAX_VALUE=16

./bin/adlalib/adla_convert --model-type onnx \
    --model onnx-3input/sensevoice_small.onnx \
    --inputs "x language text_norm" \
    --input-shapes "1,100,560#1#1" \
    --disable-per-channel False \
    --shape-with-batch "True#True#True" \
    --dtypes "float32#int32#int32" \
    --quantize-dtype int16 \
    --outdir onnx-3input/adla \
    --inference-output-type float32 \
    --target-platform PRODUCT_PID0XA005
```


| Option                   | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| `--model`                | Path to FP32 ONNX from Section 2 (`onnx-3input/sensevoice_small.onnx`) |
| `--inputs`               | Model input names: features `x`, `language`, `text_norm`               |
| `--input-shapes`         | `1,100,560` for `x`; `1` for `language`; `1` for `text_norm`           |
| `--quantize-dtype int16` | ADLA int16 quantization                                                |
| `--outdir`               | Output directory for converted `.adla` model                           |
| `--target-platform`      | SoC platform ID                                                        |



| Platform | `--target-platform` |
| -------- | ------------------- |
| S905X5   | `PRODUCT_PID0XA005` |
| A311D2   | `PRODUCT_PID0XA003` |


After conversion, the `.adla` file is generated under `onnx-3input/adla/`. Use it together with `onnx-3input/tokens.txt` from Section 2 when running the demo.

---

## 4. Demo Run

### CPP

#### 1. Compile

**Prerequisites**


| Item                                                         | Description                                                       |
| ------------------------------------------------------------ | ----------------------------------------------------------------- |
| [amlnn-toolkit](https://github.com/Amlogic-NN/amlnn-toolkit) | Set `AMLNN_HOME` or clone as sibling of this repo                 |
| Android                                                      | NDK **r25c**, set `ANDROID_NDK_PATH`                              |
| Linux (64-bit)                                               | `aarch64-linux-gnu-gcc` or ARM GNU toolchain via `GCC_COMPILER`   |
| Linux (32-bit)                                               | `arm-linux-gnueabihf-gcc` or ARM GNU toolchain via `GCC_COMPILER` |
| Yocto (optional)                                             | Yocto SDK (Poky), set `YOCTO_SDK_ROOT` or pass `-s`               |


**AMLNN setup**

```bash
git clone https://github.com/Amlogic-NN/amlnn-toolkit.git ../amlnn-toolkit
export AMLNN_HOME=/path/to/amlnn-toolkit
```

**Android (arm64-v8a / armeabi-v7a)**

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit ./build-android.sh -a arm64-v8a
```

Output: `cpp/build/android/sensevoice_demo`

**Linux 64-bit (aarch64)**

```bash
cd examples/sensevoice/cpp
export GCC_COMPILER=/path/to/toolchain/bin/aarch64-none-linux-gnu   # if not in PATH
AMLNN_HOME=/path/to/amlnn-toolkit ./build-linux.sh -m linux -a aarch64
```

Output: `cpp/build/linux/aarch64/sensevoice_demo`

**Linux 32-bit (armhf)**

```bash
cd examples/sensevoice/cpp
export GCC_COMPILER=/path/to/toolchain/bin/arm-none-linux-gnueabihf   # if not in PATH
AMLNN_HOME=/path/to/amlnn-toolkit ./build-linux.sh -m linux -a armhf
```

Output: `cpp/build/linux/armhf/sensevoice_demo`

**Yocto**

```bash
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit ./build-linux.sh -m yocto -s /path/to/poky/sdk

# 32-bit Yocto
AMLNN_HOME=/path/to/amlnn-toolkit ./build-linux.sh -m yocto -b 32 -s /path/to/poky/32bit-sdk
```

Output: `cpp/build/yocto/64/sensevoice_demo` or `cpp/build/yocto/32/sensevoice_demo`

> **Note:** The executable targets the board architecture. It **cannot** be run on the host PC in the build directory. Push it to the device via `adb` before running.

#### 2. Run

```bash
cd examples/sensevoice

# Push executable, converted model, tokens, and test wav to device
adb push cpp/build/android/sensevoice_demo /data/local/tmp/
adb push /path/to/your_model.adla /data/local/tmp/
adb push /path/to/onnx-3input/tokens.txt /data/local/tmp/
adb push input/en.wav /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x sensevoice_demo
export LD_LIBRARY_PATH=/vendor/lib64   # or /vendor/lib for 32-bit

./sensevoice_demo \
  --model your_model.adla \
  --tokens tokens.txt \
  --lang en \
  --wav en.wav
```

**Notes**

- Replace `/path/to/your_model.adla` with the `.adla` file from Section 3.
- Replace `/path/to/onnx-3input/tokens.txt` with the `tokens.txt` from Section 2.
- For **Linux / Yocto** builds, replace the executable path (e.g. `cpp/build/linux/aarch64/sensevoice_demo`).
- If `libnnsdk.so` is not on the device, push it from `amlnn-toolkit` or ensure it is in `LD_LIBRARY_PATH`.

**Argument descriptions**


| Argument   | Description                                                |
| ---------- | ---------------------------------------------------------- |
| `--model`  | ADLA model path on device                                  |
| `--tokens` | `tokens.txt` path on device                                |
| `--lang`   | `auto` / `zh` / `en` / `ja` / `ko` / `yue`                 |
| `--wav`    | PCM16 WAV input file (16 kHz recommended)                  |
| `--itn`    | Inverse text normalization: `0` (off, default) or `1` (on) |


### Python

**Prerequisites**

- Python 3.10+
- Packages: `numpy`, `kaldi-native-fbank`, `amlnnlite`

**Install dependencies (on device)**

```bash
pip install numpy kaldi-native-fbank amlnnlite-1.0.0-cp310-cp310-linux_aarch64.whl
```

**Run on device**

Push the converted `.adla`, `tokens.txt`, script, and test wav to the board, then:

```bash
cd examples/sensevoice/py

python sensevoice.py \
  --model-path /path/to/your_model.adla \
  --tokens /path/to/onnx-3input/tokens.txt \
  --lang en \
  --wav ../input/en.wav
```

**Argument descriptions**


| Argument       | Description                                |
| -------------- | ------------------------------------------ |
| `--model-path` | Path to `.adla` model                      |
| `--tokens`     | Path to `tokens.txt`                       |
| `--wav`        | PCM16 WAV input file                       |
| `--lang`       | `auto` / `zh` / `en` / `ja` / `ko` / `yue` |
| `--itn`        | Inverse text normalization: `0` or `1`     |


---

## 5.Results

**Recognition output**

```
language: <|en|>
emotion:  <|NEUTRAL|>
event:    <|Speech|>
itn:      <|woitn|>
text:     ...
```

**Test audio**

Sample files under `input/`:


| File      | Language  |
| --------- | --------- |
| `en.wav`  | English   |
| `zh.wav`  | Chinese   |
| `ja.wav`  | Japanese  |
| `ko.wav`  | Korean    |
| `yue.wav` | Cantonese |


**Integration**

For embedding the engine in your own application, see the C API header: `cpp/src/sense_voice.h`.

---

## Quick Reference (full workflow)

```text
1. Export ONNX     →  SenseVoice repo: python export-onnx-3input.py
                       → onnx-3input/sensevoice_small.onnx
                       → onnx-3input/tokens.txt

2. Convert ADLA    →  adla_convert ... --model onnx-3input/sensevoice_small.onnx
                       --target-platform PRODUCT_PID0XA005
                       → onnx-3input/adla/*.adla

3. Compile demo    →  ./build-android.sh  or  ./build-linux.sh

4. Run on board    →  adb push demo + *.adla + tokens.txt + wav
                       → ./sensevoice_demo --model ... --tokens ... --wav ...
```


# SenseVoice

This example runs SenseVoice multilingual speech recognition with AMLNN. The full flow is:

1. Prepare or export an ONNX model.
2. Convert the ONNX model to an ADLA model.
3. Run the Python or C++ demo with the ADLA model.
4. Check recognition results and profiling reports.

## Directory Layout

```text
examples/sensevoice/
|-- cpp/                  # C++ demo and build scripts
|-- input/                # Input wav files for demo (en/zh/ja/ko/yue)
|-- model/                # Put ONNX and ADLA models here
|-- py/                   # Python conversion and demo scripts
`-- README.md
```



## License

This example code is licensed under the Apache License 2.0. See the repository root [LICENSE](../../LICENSE) file for details.

The SenseVoice model is provided by a third party (FunAudioLLM). Check and follow the license and usage terms from the model source before redistribution or commercial use.

The pre-processing and post-processing code in this example follows the public kaldi-native-fbank and SenseVoice reference. If any part is similar to existing work and causes concern, please contact us and we will remove or adjust it.

## 1. Prepare The ONNX Model

Model source:

[https://github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)

The upstream SenseVoice repo provides an `export.py` that produces a **4-input ONNX** with explicit `speech`, `speech_lengths`, `language`, `textnorm` tensors. That 4-input ONNX is **not** directly usable on ADLA — this example therefore ships an adapted export script that fixes the audio length (`speech_lengths`) as a graph constant and drops it from the input list, yielding a **3-input ONNX** variant with explicit `x`, `language`, `text_norm` tensors, which ADLA tooling can ingest.

### Export ONNX

Install or clone the SenseVoice model package:

```bash
git clone https://github.com/FunAudioLLM/SenseVoice.git
```

Copy the adapted 3-input export script from this example into the SenseVoice repo, then run it instead of the upstream `export.py`:

```bash
cd SenseVoice
pip install -r requirements.txt
pip install onnx onnxruntime onnxscript
cp /path/to/amlnn-model-playground/examples/sensevoice/py/export-onnx-3input.py .
python export-onnx-3input.py
```

The script writes `onnx-3input/` containing the FP32 3-input ONNX model, the dynamic-quantized variant, and `tokens.txt`. On first run it downloads `iic/SenseVoiceSmall` weights from ModelScope (network required).

Place the generated `onnx-3input/sensevoice_small.onnx` and `onnx-3input/tokens.txt` under `examples/sensevoice/model/` for the next step:

```text
examples/sensevoice/model/sensevoice_small.onnx
examples/sensevoice/model/tokens.txt
```



## 2. Convert ONNX To ADLA

Run the ADLA export script from `examples/sensevoice/py`:

```bash
cd examples/sensevoice/py
python export_adla.py \
  --onnx ../model/sensevoice_small.onnx \
  --target-platform 007 \
  --adla ../model/SenseVoice_w8a16_A311Y3.adla
```

Arguments:


| Argument            | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `--onnx`            | Path to the SenseVoice 3-input FP32 ONNX model.                  |
| `--target-platform` | Platform ID. A311D2: 003. S905X5: 005. C302X2: 006. A311Y3: 007. |
| `--adla`            | Optional output ADLA path (default: `../model`).                 |


> Calibration dataset is not provided here; `compile()` runs with random calibration data. Accuracy may degrade — prepare an NPY calibration dataset (recommended 100-300 groups covering all three inputs `x`, `language`, `text_norm`) for production use.

After conversion, the expected model path is:

```text
examples/sensevoice/model/SenseVoice_w8a16_A311Y3.adla
```



## 3. Run Python Demo



### Prerequisites

- Python 3.10
- `numpy`
- `kaldi-native-fbank`
- AMLNN Python wheel for the target device

Install dependencies on the target device:

```bash
pip install numpy kaldi-native-fbank amlnn_edge_toolkit-1.0.0-cp310-cp310-linux_aarch64.whl
```

Run inference:

```bash
cd examples/sensevoice/py
python sensevoice_inference.py \
  --model ../model/SenseVoice_w8a16_A311Y3.adla \
  --tokens ../model/tokens.txt \
  --wav ../input/en.wav \
  --lang en
```

Python demo arguments:


| Argument   | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| `--model`  | Path to the ADLA model.                                     |
| `--tokens` | Path to `tokens.txt`.                                       |
| `--wav`    | PCM16 WAV input file (16 kHz mono recommended).             |
| `--lang`   | `auto` / `zh` / `en` / `ja` / `ko` / `yue`.                 |
| `--itn`    | Inverse text normalization: `0` (off, default) or `1` (on). |




## 4. Run C++ Demo



### Build For Android

Prerequisites:

- Android NDK, r25e recommended
- `ANDROID_NDK_PATH`, `ANDROID_NDK`, or `ANDROID_NDK_HOME` set
- `AMLNN_HOME` pointing to the amlnn_runtime
- `kaldi-native-fbank` (fetched automatically by CMake)

Build:

```bash
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime ./build-android.sh -a arm64-v8a
```

The executable is generated at:

```text
examples/sensevoice/cpp/build/android/sensevoice_demo
```



### Build For Linux

The Linux build supports two modes: **standard cross-compilation** (default) and **Yocto SDK compilation**.

#### Mode 1: Standard Linux Cross-Compile (Default)

**Prerequisites:**

- A GCC cross-compiler toolchain (GCC 10.3+ recommended)
- The toolchain `bin/` directory added to `PATH`
- `AMLNN_HOME` environment variable set
- `kaldi-native-fbank` (fetched automatically by CMake)

**1. Setup environment:**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# For 32-bit (arm) builds:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

**2. Build:**

```bash
cd examples/sensevoice/cpp

# Build for 64-bit (default)
./build-linux.sh

# Or explicitly:
./build-linux.sh -b 64
./build-linux.sh -a aarch64

# Build for 32-bit
./build-linux.sh -b 32
./build-linux.sh -a armhf
```

If your compiler uses a different prefix, override it with `GCC_COMPILER`:

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

Build outputs:

```text
build/linux/64/sensevoice_demo
build/linux/32/sensevoice_demo
```



#### Mode 2: Yocto Build

**Prerequisites:**

- Yocto SDK installed
- CMake toolchain file available

**Build:**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/sensevoice/cpp

# Build for Yocto 64-bit (default)
./build-linux.sh -m yocto -b 64 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

You can also set `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` instead of passing `-s` and `-t`.

Build outputs:

```text
build/yocto/64/sensevoice_demo
build/yocto/32/sensevoice_demo
```



### Run On Device



#### Android

Push files:

```bash
adb push build/android/sensevoice_demo /data/local/tmp/
adb push ../model/SenseVoice_w8a16_A311Y3.adla /data/local/tmp/
adb push ../model/tokens.txt /data/local/tmp/
adb push ../input/en.wav /data/local/tmp/
```

Run:

```bash
adb shell
cd /data/local/tmp
chmod +x sensevoice_demo
export LD_LIBRARY_PATH=/vendor/lib64

./sensevoice_demo \
  --model SenseVoice_w8a16_A311Y3.adla \
  --tokens tokens.txt \
  --lang en \
  --wav en.wav
```



#### Linux

Copy the binary, model, tokens, and wav file to the target device, then run:

```bash
./sensevoice_demo \
  --model SenseVoice_w8a16_A311Y3.adla \
  --tokens tokens.txt \
  --lang en \
  --wav en.wav
```

Usage:

```text
./sensevoice_demo --model <model.adla> --tokens <tokens.txt> --lang <auto|zh|en|ja|ko|yue> --wav <input.wav> [--itn 0|1]
```



## 5. Results



### Recognition Output

The demo prints the recognition tags and transcript:

```text
language: <|en|>
emotion:  <|NEUTRAL|>
event:    <|Speech|>
itn:      <|woitn|>
text:     ...
```



### Test Audio

Sample wav files under `input/`:


| File      | Language  |
| --------- | --------- |
| `en.wav`  | English   |
| `zh.wav`  | Chinese   |
| `ja.wav`  | Japanese  |
| `ko.wav`  | Korean    |
| `yue.wav` | Cantonese |




### Integration

For embedding the engine in your own application, see the C API header: `cpp/src/sense_voice.h`.
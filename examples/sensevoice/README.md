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

https://github.com/FunAudioLLM/SenseVoice

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
pip install onnx onnxruntime
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

| Argument | Description |
| --- | --- |
| `--onnx` | Path to the SenseVoice 3-input FP32 ONNX model. |
| `--target-platform` | Platform ID. A311D2: 003. S905X5: 005. C302X2: 006. A311Y3: 007. |
| `--adla` | Optional output ADLA path (default: `../model`). |

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
pip install numpy kaldi-native-fbank amlnn-1.0.0-cp310-cp310-linux_aarch64.whl
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

| Argument | Description |
| --- | --- |
| `--model` | Path to the ADLA model. |
| `--tokens` | Path to `tokens.txt`. |
| `--wav` | PCM16 WAV input file (16 kHz mono recommended). |
| `--lang` | `auto` / `zh` / `en` / `ja` / `ko` / `yue`. |
| `--itn` | Inverse text normalization: `0` (off, default) or `1` (on). |

## 4. Run C++ Demo

### Build For Android

Prerequisites:

- Android NDK, r25e recommended
- `ANDROID_NDK_PATH`, `ANDROID_NDK`, or `ANDROID_NDK_HOME` set
- `AMLNN_HOME` pointing to the AMLNN toolkit
- `kaldi-native-fbank` (fetched automatically by CMake)

Build:

```bash
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit ./build-android.sh -a arm64-v8a
```

The executable is generated at:

```text
examples/sensevoice/cpp/build/android/sensevoice_demo
```

### Build For Linux / Yocto

Prerequisites:

- A `aarch64-linux-gnu` (default) or `arm-linux-gnueabihf` cross-compiler in `PATH`
- `AMLNN_HOME` pointing to the AMLNN toolkit
- `kaldi-native-fbank` (fetched automatically by CMake)
- For Yocto: `YOCTO_SDK_ROOT` set, or pass `-s <sdk_root>`

Build (Linux, aarch64):

```bash
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit ./build-linux.sh -m linux -a aarch64
```

Build (Linux, armhf 32-bit):

```bash
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit ./build-linux.sh -m linux -a armhf
```

Build (Yocto, 64-bit):

```bash
cd examples/sensevoice/cpp
AMLNN_HOME=/path/to/amlnn-toolkit \
YOCTO_SDK_ROOT=/path/to/poky \
./build-linux.sh -m yocto -b 64
```

The executable is generated at:

```text
examples/sensevoice/cpp/build/linux/<arch>/sensevoice_demo
examples/sensevoice/cpp/build/yocto/<32|64>/sensevoice_demo
```

### Run On Device

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

| File | Language |
| --- | --- |
| `en.wav` | English |
| `zh.wav` | Chinese |
| `ja.wav` | Japanese |
| `ko.wav` | Korean |
| `yue.wav` | Cantonese |

### Integration

For embedding the engine in your own application, see the C API header: `cpp/src/sense_voice.h`.
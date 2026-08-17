# Wav2Vec2

This example runs Wav2Vec2 automatic speech recognition with AMLNN. The full flow is:

1. Prepare an ONNX model or download the prepared ADLA model.
2. Run the Python demo with the ADLA model.
3. Run the C++ (Linux/Android) demo with the ADLA model.
4. Check audio transcription results.

## Directory Layout

```bash
examples/wav2vec/
├── cpp/               # C++ demo and build scripts
├── input/             # Input WAV audio files for demo
├── model/             # Put ONNX and ADLA models here
└── py/                # Python conversion and demo scripts
```

## License

This example code is licensed under the Apache License 2.0. See the repository root [LICENSE](../../LICENSE) file for details.

The Wav2Vec2 model used in this example is based on the [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) checkpoint released by Facebook AI. The checkpoint is distributed under the **Apache License 2.0** and was pretrained and fine-tuned for English automatic speech recognition using 960 hours of 16 kHz LibriSpeech audio.

The original Wav2Vec 2.0 model implementation was released through the [fairseq repository](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec), which is distributed under the MIT License.

The ONNX model included with this example is a converted form of the original pretrained checkpoint. The Apache License 2.0 permits use, modification, and redistribution in source or compiled form, provided that its conditions are followed. When redistributing the ONNX model, retain the applicable copyright and attribution notices, include a copy of the Apache License 2.0, identify any modifications, and preserve any applicable `NOTICE` information supplied with the original work.

The 32-token CTC vocabulary used by this example is derived from the `vocab.json` file supplied with the original `facebook/wav2vec2-base-960h` checkpoint. This example does not include or redistribute the LibriSpeech dataset. Any separately obtained audio or dataset files remain subject to their own license terms.

Parts of the model preprocessing and postprocessing code in this example were developed with AI assistance. Please report any suspected licensing or attribution issue so that it can be reviewed and corrected.

## 1. Prepare The ADLA Model

### Download ADLA

Download the prepared ADLA model:

### [Download Wav2Vec2 ADLA model here!](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/model_zoo/wav2vec2/wav2vec2_base_960h_20s.adla)

Place the downloaded model under:

```text
examples/wav2vec/model/wav2vec2_base_960h_20s.adla
```

<!-- ## 2. Convert ONNX To ADLA

If you are converting the ONNX model instead of using the prepared ADLA model, run the ADLA export script from `examples/wav2vec/py`:

```bash
cd examples/wav2vec/py
python export_adla.py \
  --onnx ../model/wav2vec2_base_960h_20s.onnx \
  --target-platform 007 \
  --output-dir ../model
```

| Parameter           | Description                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `--onnx`            | Path to the input `.onnx` model.                                                                |
| `--target-platform` | Target platform ID. See the full list of supported platforms [**HERE**](../../docs/mapping.md). |
| `--output-dir`      | (Optional) Directory where the generated `.adla` model will be saved. Defaults to `../model`.   |

After conversion, the generated filename from AMLNN is preserved. For the example `wav2vec.onnx` with the current `w16a16` configuration, the expected model path is:

```text
examples/wav2vec/model/wav2vec_w16a16.adla
```

The current conversion configuration uses `w16a16` with `activation_dtype="f16"`, so no quantization dataset argument is required. -->

## 2. Run Python Demo

### Prerequisites

* Python 3.10
* Required packages: `amlnn`, `librosa`

### Install Dependencies

```bash
pip install amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl librosa
```

### Run on Device

```bash
python wav2vec_inference.py \
    --adla ../model/wav2vec2_base_960h_20s.adla \
    --audio-dir ../input
```

Argument Descriptions:

| Argument       | Description                              |
| -------------- | ---------------------------------------- |
| `--adla` | Path to the Wav2Vec2 `.adla` model.      |
| `--audio-dir`  | Directory containing `.wav` audio files. |

The script will automatically process `.wav` audio files in the specified audio directory. Audio is split into segments as required by the model, and the concatenated transcription is printed to the console.

## 3. Run C++ Demo

### Build For Android

#### Prerequisites

* **Android NDK** (r27d recommended) installed on your system.
* **AMLNN Toolkit** downloaded and extracted.

#### 1. Setup Environment

Export the paths to your NDK (the toolchain) and AMLNN (the neural network dependency) so the script can find them.

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r27d
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

#### 2. Build

Navigate to the C++ directory and run the build script.

```bash
cd examples/wav2vec/cpp

# Build for 64-bit (arm64-v8a) - Default
./build-android.sh

# Build for 32-bit (armeabi-v7a)
./build-android.sh -a armeabi-v7a
```

The executable will be generated in the build folder corresponding to the selected Android ABI:

* 64-bit: `build/android/arm64-v8a/wav2vec_demo`
* 32-bit: `build/android/armeabi-v7a/wav2vec_demo`

#### 3. Example Run

The following example uses the default 64-bit (`arm64-v8a`) build.

```bash
# Push executable and assets to device
adb shell "mkdir -p /data/local/tmp/"
adb push build/android/arm64-v8a/wav2vec_demo /data/local/tmp/
adb push ../model/wav2vec2_base_960h_20s.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x wav2vec_demo
export LD_LIBRARY_PATH=/vendor/lib64

# Usage: ./wav2vec_demo <model_path> <audio_dir>
./wav2vec_demo wav2vec2_base_960h_20s.adla input/
```

> **Note:** For a 32-bit (`armeabi-v7a`) build, use `build/android/armeabi-v7a/wav2vec_demo` and the corresponding 32-bit library path. Replace `wav2vec2_base_960h_20s.adla` with your actual model file name.

---

### Build For Linux

The Linux build process supports two distinct modes:

1. **Standard Linux cross-compilation** (default)
2. **Yocto SDK compilation**

#### Mode 1: Standard Linux Cross-Compile (Default)

##### Prerequisites

* A GCC Cross-Compiler toolchain (GCC 10.3 recommended).
* The toolchain's `bin/` folder must be added to your system's `PATH`.
* `AMLNN_HOME` environment variable set.

##### 1. Setup Environment

Add your downloaded toolchain to your `PATH` and export the `AMLNN_HOME` variable so the script can find the compiler and neural network dependencies.

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds, add the 64-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# OR for 32-bit (arm) builds, add the 32-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

##### 2. Build

```bash
cd examples/wav2vec/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

> **Optional Override:** If your compiler has a different prefix name, for example `aarch64-linux-gnu` instead of `aarch64-none-linux-gnu`, you can override the default by setting the `GCC_COMPILER` variable:

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/linux/64/wav2vec_demo`
* 32-bit: `build/linux/32/wav2vec_demo`

#### Mode 2: Yocto Build

##### 1. Prerequisites

* Yocto SDK installed.
* CMake Toolchain file available.

##### 2. Setup Environment

```bash
# Export the AMLNN path
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime
```

##### 3. Build

```bash
cd examples/wav2vec/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

> **Note:** You can also use the `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` environment variables instead of passing the `-s` and `-t` flags.

The executable will be generated in the build folder corresponding to the selected architecture:

* 64-bit: `build/yocto/64/wav2vec_demo`
* 32-bit: `build/yocto/32/wav2vec_demo`

#### 3. Example Run

The following example uses the default 64-bit Linux build.

```bash
# Push executable and assets to device (adjust build path if using Yocto)
adb shell "mkdir -p /data/local/tmp/"
adb push build/linux/64/wav2vec_demo /data/local/tmp/
adb push ../model/wav2vec2_base_960h_20s.adla /data/local/tmp/
adb push ../input/ /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
chmod +x wav2vec_demo

# Usage: ./wav2vec_demo <model_path> <audio_dir>
./wav2vec_demo wav2vec2_base_960h_20s.adla input/
```

> **Note:** Replace `wav2vec2_base_960h_20s.adla` with your actual model file name. Adjust the executable path if using a 32-bit or Yocto build.

## 4. Results

### Performance Feedback

By setting the log level to `INFO`, the program provides runtime performance information after inference. The console output may include:

* **Hardware Information:** System and ADLA library versions.
* **Model Overview:** Input and output tensor configurations.
* **NPU Metrics:** Inference latency and DRAM bandwidth usage.

### Audio Transcription Output

For each `.wav` audio file, the program splits the audio into segments as required by the model, processes each segment, and prints the concatenated transcription to the console.

Example output:

```text
============================================================
Processing [1/2]: Alexanders_Bridge_Chapter_II_by_Willa_Cather.wav
============================================================
Segments: 8
Processing segment [1/8]...
Processing segment [2/8]...
Processing segment [3/8]...
Processing segment [4/8]...
Processing segment [5/8]...
Processing segment [6/8]...
Processing segment [7/8]...
Processing segment [8/8]...
Transcription: MAIN HALL LIKED ALEXANDER BECAUSE HE WAS AN ENGINEER HE HAD PRECONCEIVED IDEAS ABOUT EVERYTHING AND HIS IDEA ABOUT AMERICANS WAS THAT THEY SHOULD BE ENGINEERS ON MECHANICS IT'S TREMENDOUSLY WELL PUT ON TOO IT'S BEEN ON ONLY TWO WEEKS AND I'VE BEEN HALF A DOZEN TIMES ALREADY DO YOU KNOW ALEXANDER MAIN HLL LOOKED WITH PERPLEXITY UP INTO THE TOP OF THE HANSOM AND RUBBED HIS PINK CHEEK WITH HIS GLOVED FINGER DOYOU KNOW I SOMETIMES THINK OF TAKING TO CRITICISM SERIOUSLY MYSELF SHE SAVES HER HAND TOO SHE'S AT HER BEST IN THE SECONDACT HE'S BEEN WANTING TO MARRY HILDA THESE THREE YEARS AND MORE SHE DOESN'T TAKE UP WITH ANYBODY YOU KNOW IREN BERGOING ONE OF HER FAMILY TOLD ME IN CONFIDENCE THAT THERE WAS A ROMANCE SOMEWHERE BACK IN THE BEGINNING MAINHALL VOUCHED FOR HER CONSTANCY WITH A LOFTINESS THAT MADE ALEXANDER SMILE EVEN WHILE A KIND OF RAPID EXCITEMENT WAS TINGLING THROUGH HIM HE'S ANOTHER WHO'S AWFULLY KEEN ABOUT HER LET ME INTRODUCE YOU SIR HARRY TOWN MISTER BARTLEY ALEXANDER THE AMERICAN ENGINEER I SAY SIR HARRY THE LITTLE GIRL'S GOING FAMOUSLY TO NIGHT ISN'T SHE YOU KNOW I THOUGHT THE DANCE AF BI CONSCIENCE TO NIGHT FOR THE FIRST TIME WESTMERE AND I WERE BACK AFTER THE FIRST ACT AND WE THOUGHT SHE SEEMED QUITE UNCERTAIN OF HERSELF A LITTLE ATTACK OF NERVES POSSIBLY HE WAS BEGINNING TO FEEL KEEN INTEREST IN THE SLENDER BAREFOOT DONKEY GIRL WHO SLIPPED IN AND OUT OF THE PLAY SINGING LIKE SOME ONE WINDING THROUGH A HLLY FIELD ONE NIGHT WHEN HE AND WINTIFORD WERE SITTING TOGETHER ON THE BRIDGE HE TOLD HER THAT THINGS HAD HAPPENED WHILE HE WAS STUDYING ABROAD THAT HE WAS SORRY FOR ONE THING IN PARTICULAR AND HE ASKED HER WHETHER SHE THOUGHT SHE OUGHT TO KNOW ABOUT THEM SHE CONSIDERED FOR A MOMENT AND THEN SAID NO I THINK NOT THOUGH I AM GLAD YOU ASK ME AFTER THAT IT WAS EASY TO FORGET ACTUALLY TO FORGET OF COURSE HE REFLECTED SHE ALWAYS HAD THAT COMBINATION OF SOMETHING HOMELY AND SENSIBLE AND SOMETHING UTTERLY WILD AND DAFT SHE MUST CARE ABOUT THE THEATRE A GREAT DEAL MORE THAN SHE USED TO I'M GLAD SHE'S HELD HER OWN SINCE AFTER ALL WE WERE AWFULLY YOUNG I SHOULDN'T WONDER IF SHE COULD LAUGH ABOUT IT WITH ME NOW
============================================================
```

## Third Party Notices

The example audio file is derived from the LibriSpeech ASR corpus, created by Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur.

LibriSpeech is distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Source: OpenSLR SLR12, LibriSpeech `test-clean`, speaker/chapter `4446/2271`.

Original files: `4446-2271-0000.flac` through `4446-2271-0024.flac`.

Modifications: The original LibriSpeech utterances were concatenated, converted from FLAC to WAV, and resampled for use as an example input. No endorsement by the LibriSpeech authors or contributors is implied.

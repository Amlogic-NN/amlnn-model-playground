# DTLN

This example demonstrates how to run the [DTLN](https://github.com/breizhn/DTLN) speech enhancement model with AMLNN.

The full process is:

1. Download or prepare the ONNX models.
2. Convert the ONNX models to ADLA format.
3. Build and run the C++ demo with the ADLA models.
4. Retrieve the enhanced output audio.

## Directory Layout

```text
examples/dtln/
|-- cpp/                  # C++ demo source and build scripts
|-- model/                # Place ONNX and ADLA models here
|-- in.wav                # Example input audio
|-- out.wav               # Example output audio
```

## License

This example code is licensed under the Apache License 2.0. See the repository root [LICENSE](../../LICENSE) file for details.

## 1. Prepare the ONNX Models

The DTLN example uses two pretrained ONNX models:

- [model_1.onnx](https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_1.onnx)
- [model_2.onnx](https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_2.onnx)

Download both files into the `examples/dtln/model/` directory.

## 2. Convert ONNX to ADLA

Use the provided conversion script to generate ADLA models from the ONNX files.

```bash
python ./py/export_adla.py --model1 ./model/model_1.onnx --model2 ./model/model_2.onnx --target-platform 007
```

After conversion, place the output ADLA files under `examples/dtln/model/`.

## 3. Run Python Demo

## 4. Build the C++ Demo

### Build for Android

**Prerequisites:**

- Android NDK (r25e recommended)

**Build:**

Build the Android demo from `examples/dtln/cpp/`.

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25e
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/dtln/cpp
./build-android.sh
```

The resulting executable is located at `examples/dtln/cpp/build/android/dtln_demo`.

**Run:**

Copy the demo binary, model files, and input audio to the device.

```bash
adb push examples/dtln/cpp/build/android/dtln_demo /data/local/tmp/
adb push examples/dtln/model/dtln_model_1.adla examples/dtln/model/dtln_model_2.adla /data/local/tmp/
adb push examples/dtln/in.wav /data/local/tmp/
```

Run the demo on the device:

```bash
adb shell
cd /data/local/tmp
chmod +x dtln_demo
export LD_LIBRARY_PATH=/vendor/lib64
./dtln_demo ./dtln_model_1.adla ./dtln_model_2.adla ./in.wav ./out.wav
```

### Build for Linux

The Linux build process supports two distinct modes: **Standard Linux cross-compilation** (default) and **Yocto SDK compilation**.

### Mode 1: Standard Linux Cross-Compile (Default)

**Prerequisites:**

- A GCC Cross-Compiler toolchain (GCC 10.3 recommended).
- The toolchain's `bin/` folder must be added to your system's `PATH`.
- `AMLNN_HOME` environment variable set

**Setup Environment:**

Add your downloaded toolchain to your `PATH` and export the `AMLNN_HOME` variable so the script can find the compiler and neural network dependencies.

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

# For 64-bit (aarch64) builds, add the 64-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:$PATH

# OR for 32-bit (arm) builds, add the 32-bit toolchain to PATH:
export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf/bin:$PATH
```

**Build:**

```bash
cd examples/dtln/cpp

# Build for 64-bit (Default)
./build-linux.sh

# Build for 32-bit
./build-linux.sh -b 32
```

*(Optional Override):* If your compiler has a different prefix name (for example, `aarch64-linux-gnu` instead of `aarch64-none-linux-gnu`), you can override the default by setting the `GCC_COMPILER` variable:

```bash
GCC_COMPILER=aarch64-linux-gnu ./build-linux.sh
```

The executable will be generated at `build/linux/64/dtln_demo` (or `build/linux/32/dtln_demo`).

### Mode 2: Yocto Build

**Prerequisites:**

- Yocto SDK installed
- CMake Toolchain file available

**Build:**

```bash
export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime

cd examples/dtln/cpp

# Build for Yocto 64-bit (Default)
./build-linux.sh -m yocto -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake

# Or build for Yocto 32-bit
./build-linux.sh -m yocto -b 32 -s /path/to/yocto_sdk_root -t /path/to/toolchain.cmake
```

*(Note: You can also use the `YOCTO_SDK_ROOT` and `TOOLCHAIN_FILE` environment variables instead of passing the `-s` and `-t` flags).*

The executable will be generated at `build/yocto/64/dtln_demo` (or `build/yocto/32/dtln_demo`).

---

**Run:**

The runtime procedure is identical to the Android section above.

## 5. Result

The demo writes `out.wav` to the device directory.

Example result:

- [in.wav](./in.wav)
- [out.wav](./out.wav)
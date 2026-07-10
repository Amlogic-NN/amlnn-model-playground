# YOLOX Android JNI Demo

This example runs YOLOX object detection on Amlogic NPU via **AMLNN SDK2** and a JNI bridge. The app is built with Kotlin + Jetpack Compose and loads the AMLNN runtime through `libnnsdk.so`.

Supported ABIs: **arm64-v8a** (64-bit) and **armeabi-v7a** (32-bit).

## Directory Layout

```text
examples/yolox/jni/
├── app/
│   ├── src/main/
│   │   ├── assets/demo/          # Model and test images (see below)
│   │   ├── java/com/example/NN_JNI/
│   │   ├── jni/                  # nnsdk_jni.cpp, nnsdk2.h
│   │   └── libs/
│   │       ├── arm64-v8a/        # Prebuilt AMLNN runtime (64-bit)
│   │       └── armeabi-v7a/      # Prebuilt AMLNN runtime (32-bit)
│   ├── CMakeLists.txt
│   └── build.gradle.kts
├── gradle/
├── build.gradle.kts
├── settings.gradle.kts
└── README.md
```

## Prerequisites

- Android Studio (Ladybug or newer recommended)
- Android SDK (API 35)
- Android NDK **r25c** or **r25e** recommended  
  Download: https://github.com/android/ndk/wiki/Unsupported-Downloads
- Amlogic device with ADLA NPU support (e.g. A311D2, S905X5)

## Prepare Demo Assets

Before building, place the following files under `app/src/main/assets/demo/`:

| File | Description |
| ---- | ----------- |
| `yolox_s_int8.adla` | Quantized YOLOX-S ADLA model |
| `*.jpg` / `*.png` | Optional test images |

To obtain the ADLA model, follow the conversion steps in the parent [YOLOX README](../README.md) (Section 1). You can also download a pre-built model from the [Amlogic model zoo](https://huggingface.co/Amlogic-NN/amlnn-adla-models/tree/main).

Rename or export the model to match `yolox_s_int8.adla`, or update `DEFAULT_MODEL_NAME` in `DemoAssetHelper.kt`.

COCO class names are built into `CocoLabels.kt`; no separate labels file is required.

## Build

### Option A: Android Studio

1. Open the `jni/` directory in Android Studio.
2. Create `local.properties` with your SDK path:

   ```properties
   sdk.dir=/path/to/Android/Sdk
   ```

3. Build → **Build APK(s)** or run on a connected device.

### Option B: Command Line

```bash
cd examples/yolox/jni

# Create local.properties if needed
echo "sdk.dir=/path/to/Android/Sdk" > local.properties

# Build debug APK (both 32-bit and 64-bit ABIs)
./gradlew assembleDebug
```

Output APK:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## Install and Run

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.yolox_jni/com.example.NN_JNI.MainActivity
```

On first launch the app copies assets from `assets/demo/` to the external files directory, then runs inference on the selected image. The UI draws bounding boxes and class labels on the result image.

## Native Libraries

Prebuilt `.so` files are bundled per ABI under `app/src/main/libs/`:

| Library | Role |
| ------- | ---- |
| `libnnsdk.so` | AMLNN SDK2 runtime (loaded by JNI) |
| `libadla.so` | ADLA NPU driver (preloaded before nnsdk) |
| `libc++.so` | LLVM libc++ runtime |
| `libteec.so` | OP-TEE client library |

`nnsdk_jni` (built from `app/src/main/jni/nnsdk_jni.cpp` via CMake) dynamically loads `libnnsdk.so` at runtime. Gradle packages the matching ABI libraries into the APK automatically.

To refresh these libraries from a local SDK checkout:

```bash
# Example: copy from amlnn-toolkit (adjust paths as needed)
AMLNN_HOME=/path/to/amlnn-toolkit
cp $AMLNN_HOME/android/arm64-v8a/*.so   app/src/main/libs/arm64-v8a/
cp $AMLNN_HOME/android/armeabi-v7a/*.so app/src/main/libs/armeabi-v7a/
```

See the repository root [README](../../../README.md) for `amlnn-toolkit` setup.


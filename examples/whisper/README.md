# Whisper

## 4. Demo Run

### CPP

#### 1. Compile

**Prerequisites:**
- Android NDK r25c
- `ANDROID_NDK_PATH` environment variable set

**Build:**
```bash
# Build for arm64-v8a
cd examples/whisper/cpp
AMLNN_HOME=/path/to/amlnn-toolkit ./build-android.sh -a arm64-v8a
```

The executable will be generated in `build/android/`.

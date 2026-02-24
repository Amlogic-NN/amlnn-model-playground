# YOLOE

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
- Android NDK r25c
- `ANDROID_NDK_PATH` environment variable set

**Build:**
```bash
# Build for arm64-v8a
cd examples/yoloe/cpp
./build-android.sh -a arm64-v8a
```

The executable will be generated in `build/android/`.

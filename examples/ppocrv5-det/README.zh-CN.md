[EN](README.md) | **CN**

# PaddleOCR 文本检测

## 4. 运行示例

### CPP 示例

#### 1. 编译

**前置条件：**
- Android NDK r25c 版本
- 需要提前设置 `ANDROID_NDK_PATH` 环境变量

**构建步骤：**
```bash
# Build for arm64-v8a
cd examples/ppocr-det/cpp
AMLNN_HOME=/path/to/amlnn-toolkit ./build-android.sh -a arm64-v8a
```

编译完成后，生成的可执行文件将保存在 `build/android/` 目录中。

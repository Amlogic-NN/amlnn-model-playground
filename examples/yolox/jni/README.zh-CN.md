[EN](README.md) | **CN**

# YOLOX Android JNI 示例

本示例通过 **AMLNN SDK2** 与 JNI 桥接在 Amlogic NPU 上运行 YOLOX 目标检测。应用使用 Kotlin + Jetpack Compose 构建，并通过 `libnnsdk.so` 加载 AMLNN 运行时。

支持的 ABI：**arm64-v8a**（64 位）和 **armeabi-v7a**（32 位）。

## 目录结构

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

## 前置条件

- Android Studio（建议 Ladybug 或更新版本）
- Android SDK（API 35）
- 建议使用 Android NDK **r25c** 或 **r25e**  
  下载：https://github.com/android/ndk/wiki/Unsupported-Downloads
- 支持 ADLA NPU 的 Amlogic 设备（例如 A311D2、S905X5）

## 准备示例资源

构建前，请将以下文件放到 `app/src/main/assets/demo/` 下：

| 文件 | 说明 |
| ---- | ----------- |
| `yolox_s_int8.adla` | 量化后的 YOLOX-S ADLA 模型 |
| `*.jpg` / `*.png` | 可选的测试图片 |

如需获取 ADLA 模型，请按照上级 [YOLOX README](../README.zh-CN.md) 中的转换步骤操作（第 1 节）。也可以从 [Amlogic 模型库](https://huggingface.co/Amlogic-NN/amlnn-adla-models/tree/main) 下载预构建的模型。

请重命名或导出模型，使其与 `yolox_s_int8.adla` 一致；或修改 `DemoAssetHelper.kt` 中的 `DEFAULT_MODEL_NAME`。

COCO 类别名称已内置于 `CocoLabels.kt`，无需单独的标签文件。

## 构建

### 方式 A：Android Studio

1. 在 Android Studio 中打开 `jni/` 目录。
2. 创建 `local.properties` 并填入你的 SDK 路径：

   ```properties
   sdk.dir=/path/to/Android/Sdk
   ```

3. 构建 → **Build APK(s)**，或在已连接的设备上运行。

### 方式 B：命令行

```bash
cd examples/yolox/jni

# Create local.properties if needed
echo "sdk.dir=/path/to/Android/Sdk" > local.properties

# Build debug APK (both 32-bit and 64-bit ABIs)
./gradlew assembleDebug
```

输出的 APK：

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 安装与运行

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.yolox_jni/com.example.NN_JNI.MainActivity
```

首次启动时，应用会将 `assets/demo/` 中的资源复制到外部文件目录，然后对选定图片执行推理。UI 会在结果图片上绘制边界框和类别标签。

## 原生库

预构建的 `.so` 文件按 ABI 分别放在 `app/src/main/libs/` 下：

| 库 | 作用 |
| ------- | ---- |
| `libnnsdk.so` | AMLNN SDK2 运行时（由 JNI 加载） |
| `libadla.so` | ADLA NPU 驱动（在 nnsdk 之前预加载） |
| `libc++.so` | LLVM libc++ 运行时 |
| `libteec.so` | OP-TEE 客户端库 |

`nnsdk_jni`（通过 CMake 从 `app/src/main/jni/nnsdk_jni.cpp` 构建）在运行时动态加载 `libnnsdk.so`。Gradle 会自动将对应 ABI 的库打包进 APK。

如需从本地 SDK 代码库更新这些库：

```bash
# Example: copy from amlnn-toolkit (adjust paths as needed)
AMLNN_HOME=/path/to/amlnn-toolkit
cp $AMLNN_HOME/android/arm64-v8a/*.so   app/src/main/libs/arm64-v8a/
cp $AMLNN_HOME/android/armeabi-v7a/*.so app/src/main/libs/armeabi-v7a/
```

`amlnn-toolkit` 的配置方法请参见仓库根目录的 [README](../../../README.zh-CN.md)。


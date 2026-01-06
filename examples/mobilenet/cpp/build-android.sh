#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [-a <target_abi>]"
    echo "  -a <target_abi> : Target ABI (default: arm64-v8a)"
    echo "  -h              : Show this help message"
    exit 1
}

# Default values
TARGET_ABI=arm64-v8a

# Parse arguments
while getopts 'a:h' opt; do
  case "$opt" in
    a)
      TARGET_ABI=$OPTARG
      ;;
    h)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

if [ -z "${ANDROID_NDK_PATH}" ]; then
    if [ -n "${ANDROID_NDK}" ]; then
        ANDROID_NDK_PATH=${ANDROID_NDK}
    elif [ -n "${ANDROID_NDK_HOME}" ]; then
        ANDROID_NDK_PATH=${ANDROID_NDK_HOME}
    else
        echo "Error: ANDROID_NDK_PATH is not set."
        echo "Please set ANDROID_NDK_PATH to your Android NDK directory."
        exit 1
    fi
fi

ROOT_PWD=$(cd "$(dirname $0)" && pwd)
BUILD_DIR=${ROOT_PWD}/build/android

echo "Building for Android..."
echo "NDK_PATH: ${ANDROID_NDK_PATH}"
echo "TARGET_ABI: ${TARGET_ABI}"
echo "BUILD_DIR: ${BUILD_DIR}"

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../src \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=${TARGET_ABI} \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=${ROOT_PWD}/../../../dependency/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${TARGET_ABI}

make -j$(nproc)

echo "Build complete. Executable in ${BUILD_DIR}/mobilenet_v2_demo"

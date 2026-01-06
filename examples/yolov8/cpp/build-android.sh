#!/bin/bash
set -e

#
# Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

make -j4

echo "Build complete. Executable in ${BUILD_DIR}/yolov8_demo"

#!/bin/bash
set -e

#
# Copyright (C) 2026 Amlogic, Inc. All rights reserved.
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
    echo "                    Supported values:"
    echo "                      arm64-v8a"
    echo "                      armeabi-v7a"
    echo "  -h              : Show this help message"
    exit 1
}

TARGET_ABI=arm64-v8a

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

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${ROOT_PWD}/build/android

echo "Building sensevoice for Android..."
echo "NDK_PATH:   ${ANDROID_NDK_PATH}"
echo "TARGET_ABI: ${TARGET_ABI}"
echo "BUILD_DIR:  ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -Wno-dev ../../src \
    -DAMLNN_HOME=${AMLNN_HOME:-} \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=${TARGET_ABI} \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build complete."
echo "  Static library: ${BUILD_DIR}/libsensevoice.a"
echo "  Demo binary:    ${BUILD_DIR}/sensevoice_demo"

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
    echo "  -a <target_abi> : Target ABI (arm64-v8a or armeabi-v7a, default: arm64-v8a)"
    echo "  -h              : Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  1. ANDROID_NDK_PATH (or ANDROID_NDK / ANDROID_NDK_HOME) set"
    echo "  2. AMLNN SDK discoverable via cmake/FindAMLNN.cmake (or set AMLNN_HOME)"
    exit 1
}

TARGET_ABI=arm64-v8a

while getopts 'a:h' opt; do
  case "$opt" in
    a) TARGET_ABI=$OPTARG ;;
    h) usage ;;
    *) usage ;;
  esac
done

if [[ "${TARGET_ABI}" != "arm64-v8a" && "${TARGET_ABI}" != "armeabi-v7a" ]]; then
    echo "Error: unsupported ABI '${TARGET_ABI}'"
    echo "Supported: arm64-v8a, armeabi-v7a"
    exit 1
fi

if [ -z "${ANDROID_NDK_PATH}" ]; then
    if [ -n "${ANDROID_NDK}" ]; then
        ANDROID_NDK_PATH=${ANDROID_NDK}
    elif [ -n "${ANDROID_NDK_HOME}" ]; then
        ANDROID_NDK_PATH=${ANDROID_NDK_HOME}
    else
        echo "Error: ANDROID_NDK_PATH is not set."
        exit 1
    fi
fi

ROOT_PWD=$(cd "$(dirname $0)" && pwd)
BUILD_DIR="${ROOT_PWD}/build/android_${TARGET_ABI}"

echo "Building for Android..."
echo "  NDK_PATH   : ${ANDROID_NDK_PATH}"
echo "  TARGET_ABI : ${TARGET_ABI}"
echo "  AMLNN_HOME : ${AMLNN_HOME:-<auto-detect>}"
echo "  BUILD_DIR  : ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"

cmake -Wno-dev \
    -S "${ROOT_PWD}/src" \
    -B "${BUILD_DIR}" \
    -DAMLNN_HOME="${AMLNN_HOME:-}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${TARGET_ABI}" \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --config Release -j4

echo ""
echo "Build complete. Executable: ${BUILD_DIR}/mobileclip_demo"

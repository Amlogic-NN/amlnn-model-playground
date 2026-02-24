#!/bin/bash

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

SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)

echo "============================================"
echo "Building all Android examples"
echo "NDK_PATH: ${ANDROID_NDK_PATH}"
echo "TARGET_ABI: ${TARGET_ABI}"
echo "============================================"

# List all examples that have a build-android.sh
EXAMPLES=(
    "clip/cpp"
    "mobilenet/cpp"
    "ppocr-det/cpp"
    "resnet/cpp"
    "retinaface/cpp"
    "whisper/cpp"
    "yoloe/cpp"
    "yolov11/cpp"
    "yolov8/cpp"
    "yoloworld/cpp"
    "yolox/cpp"
)

FAILED=()
SUCCEEDED=()

for EXAMPLE in "${EXAMPLES[@]}"; do
    EXAMPLE_DIR="${SCRIPT_DIR}/${EXAMPLE}"
    BUILD_SCRIPT="${EXAMPLE_DIR}/build-android.sh"

    if [ ! -f "${BUILD_SCRIPT}" ]; then
        echo "WARNING: No build-android.sh found for ${EXAMPLE}, skipping."
        continue
    fi

    echo ""
    echo "--------------------------------------------"
    echo "Building: ${EXAMPLE}"
    echo "--------------------------------------------"

    if bash "${BUILD_SCRIPT}" -a "${TARGET_ABI}"; then
        SUCCEEDED+=("${EXAMPLE}")
        echo "[SUCCESS] ${EXAMPLE}"
    else
        FAILED+=("${EXAMPLE}")
        echo "[FAILED] ${EXAMPLE}"
    fi
done

echo ""
echo "============================================"
echo "Build Summary"
echo "============================================"
echo "Succeeded (${#SUCCEEDED[@]}):"
for E in "${SUCCEEDED[@]}"; do echo "  - $E"; done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed (${#FAILED[@]}):"
    for E in "${FAILED[@]}"; do echo "  - $E"; done
    exit 1
fi

echo ""
echo "All examples built successfully!"

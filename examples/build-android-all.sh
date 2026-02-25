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

SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)

# Priority 1: Environment variable (recommended)
if [ -n "$AMLNN_HOME" ]; then
    if [ ! -d "$AMLNN_HOME/nn_runtime" ]; then
        echo "Error: AMLNN_HOME is set to '$AMLNN_HOME' but nn_runtime was not found there."
        echo "Please check your AMLNN_HOME path."
        exit 1
    fi
    RUNTIME_PATH="$AMLNN_HOME/nn_runtime"
    echo "Priority 1: Using AMLNN_HOME from environment: $AMLNN_HOME"
# Priority 3: Fallback to sibling amlnn-toolkit (compatibility)
elif [ -d "${SCRIPT_DIR}/../../amlnn-toolkit/nn_runtime" ]; then
    export AMLNN_HOME="$(cd "${SCRIPT_DIR}/../../amlnn-toolkit" && pwd)"
    RUNTIME_PATH="$AMLNN_HOME/nn_runtime"
    echo "Priority 3: Using sibling amlnn-toolkit as fallback: $AMLNN_HOME"
elif [ -d "${SCRIPT_DIR}/../../amlnn-toolkit-a/nn_runtime" ]; then
    export AMLNN_HOME="$(cd "${SCRIPT_DIR}/../../amlnn-toolkit-a" && pwd)"
    RUNTIME_PATH="$AMLNN_HOME/nn_runtime"
    echo "Priority 3: Using sibling amlnn-toolkit-a as fallback: $AMLNN_HOME"
else
    echo ""
    echo "Error: AMLNN SDK not found."
    echo ""
    echo "Please do one of the following:"
    echo ""
    echo "  Option A (recommended) – set AMLNN_HOME:"
    echo "    export AMLNN_HOME=/path/to/amlnn-toolkit"
    echo "    ./build-android-all.sh"
    echo ""
    echo "  Option B – clone amlnn-toolkit as a sibling directory:"
    echo "    git clone https://github.com/Amlogic-NN/amlnn-toolkit.git ../../amlnn-toolkit"
    echo "    ./build-android-all.sh"
    echo ""
    exit 1
fi

echo "============================================"
echo "Building all Android examples"
echo "NDK_PATH: ${ANDROID_NDK_PATH}"
echo "TARGET_ABI: ${TARGET_ABI}"
echo "============================================"

# Dynamically discover all examples that have a build-android.sh
mapfile -t BUILD_SCRIPTS < <(find "${SCRIPT_DIR}" -mindepth 3 -maxdepth 3 -name "build-android.sh" | sort)

EXAMPLES=()
for script in "${BUILD_SCRIPTS[@]}"; do
    # Convert absolute path to a path relative to SCRIPT_DIR (e.g. "yolov8/cpp")
    rel=$(realpath --relative-to="${SCRIPT_DIR}" "$(dirname "$script")")
    EXAMPLES+=("$rel")
done

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

    # Clean previous build to avoid stale CMake cache
    echo "Cleaning: ${EXAMPLE_DIR}/build/android"
    rm -rf "${EXAMPLE_DIR}/build/android"

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

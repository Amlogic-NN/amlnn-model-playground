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
    echo "Usage: $0 [-m <mode>] [-b <arch_bits>] [-s <yocto_sdk_root>] [-t <toolchain_file>]"
    echo "  -m <mode>      : Build mode: 'linux' or 'yocto' (default: linux)"
    echo "  -b <arch_bits> : Arch bits for yocto mode: 32 or 64 (default: 64)"
    echo "  -s <sdk_root>  : Yocto SDK root path (overrides YOCTO_SDK_ROOT env var)"
    echo "  -t <toolchain> : CMake toolchain file (overrides TOOLCHAIN_FILE env var)"
    echo "  -h             : Show this help message"
    exit 1
}

# Default values
BUILD_MODE=linux
TARGET_ARCH=aarch64
ARCH_BITS=64
CLI_SDK_ROOT=""
CLI_TOOLCHAIN_FILE=""

# Parse arguments
while getopts 'm:b:s:t:h' opt; do
  case "$opt" in
    m)
      BUILD_MODE=$OPTARG
      ;;
    a)
      TARGET_ARCH=$OPTARG
      ;;
    b)
      ARCH_BITS=$OPTARG
      ;;
    s)
      CLI_SDK_ROOT=$OPTARG
      ;;
    t)
      CLI_TOOLCHAIN_FILE=$OPTARG
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

# ---------------------------------------------------------------------------
# AMLNN SDK discovery (same logic as build-android-all.sh)
# ---------------------------------------------------------------------------
if [ -n "$AMLNN_HOME" ]; then
    if [ ! -d "$AMLNN_HOME/nn_runtime" ]; then
        echo "Error: AMLNN_HOME is set to '$AMLNN_HOME' but nn_runtime was not found there."
        echo "Please check your AMLNN_HOME path."
        exit 1
    fi
    echo "Priority 1: Using AMLNN_HOME from environment: $AMLNN_HOME"
elif [ -d "${SCRIPT_DIR}/../../amlnn-toolkit/nn_runtime" ]; then
    export AMLNN_HOME="$(cd "${SCRIPT_DIR}/../../amlnn-toolkit" && pwd)"
    echo "Priority 3: Using sibling amlnn-toolkit as fallback: $AMLNN_HOME"
elif [ -d "${SCRIPT_DIR}/../../amlnn-toolkit-a/nn_runtime" ]; then
    export AMLNN_HOME="$(cd "${SCRIPT_DIR}/../../amlnn-toolkit-a" && pwd)"
    echo "Priority 3: Using sibling amlnn-toolkit-a as fallback: $AMLNN_HOME"
else
    echo ""
    echo "Error: AMLNN SDK not found."
    echo ""
    echo "Please do one of the following:"
    echo ""
    echo "  Option A (recommended) – set AMLNN_HOME:"
    echo "    export AMLNN_HOME=/path/to/amlnn-toolkit"
    echo "    ./build-linux-all.sh"
    echo ""
    echo "  Option B – clone amlnn-toolkit as a sibling directory:"
    echo "    git clone https://github.com/Amlogic-NN/amlnn-toolkit.git ../../amlnn-toolkit"
    echo "    ./build-linux-all.sh"
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# Demos to exclude (directory name under examples/)
# ---------------------------------------------------------------------------
EXCLUDE_DIRS=("LLMs")

is_excluded() {
    local demo="$1"
    for excl in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$demo" == *"${excl}"* ]]; then
            return 0
        fi
    done
    return 1
}

echo "============================================"
echo "Building all Linux examples"
echo "BUILD_MODE: ${BUILD_MODE}"
if [[ "${BUILD_MODE}" == "yocto" ]]; then
    echo "ARCH_BITS: ${ARCH_BITS}-bit"
    echo "YOCTO_SDK_ROOT: ${CLI_SDK_ROOT:-${YOCTO_SDK_ROOT:-/data/yuandian/tools/poky/4.0.20}}"
else
    echo "TARGET_ARCH: ${TARGET_ARCH}"
fi
echo "EXCLUDED: ${EXCLUDE_DIRS[*]}"
echo "============================================"

# Dynamically discover all examples that have a build-linux.sh
mapfile -t BUILD_SCRIPTS < <(find "${SCRIPT_DIR}" -mindepth 3 -maxdepth 3 -name "build-linux.sh" | sort)

EXAMPLES=()
for script in "${BUILD_SCRIPTS[@]}"; do
    rel=$(realpath --relative-to="${SCRIPT_DIR}" "$(dirname "$script")")
    EXAMPLES+=("$rel")
done

FAILED=()
SUCCEEDED=()
SKIPPED=()

for EXAMPLE in "${EXAMPLES[@]}"; do
    # Skip excluded demos
    if is_excluded "${EXAMPLE}"; then
        SKIPPED+=("${EXAMPLE}")
        echo ""
        echo "[SKIP] ${EXAMPLE} (excluded)"
        continue
    fi

    EXAMPLE_DIR="${SCRIPT_DIR}/${EXAMPLE}"
    BUILD_SCRIPT="${EXAMPLE_DIR}/build-linux.sh"

    if [ ! -f "${BUILD_SCRIPT}" ]; then
        echo "WARNING: No build-linux.sh found for ${EXAMPLE}, skipping."
        continue
    fi

    echo ""
    echo "--------------------------------------------"
    echo "Building: ${EXAMPLE}"
    echo "--------------------------------------------"

    # Clean previous build to avoid stale CMake cache
    if [[ "${BUILD_MODE}" == "yocto" ]]; then
        echo "Cleaning: ${EXAMPLE_DIR}/build/yocto/${ARCH_BITS}"
        rm -rf "${EXAMPLE_DIR}/build/yocto/${ARCH_BITS}"
    else
        echo "Cleaning: ${EXAMPLE_DIR}/build/linux"
        rm -rf "${EXAMPLE_DIR}/build/linux"
    fi

    # Build the extra args to forward to individual build-linux.sh
    EXTRA_ARGS=(-m "${BUILD_MODE}")
    if [[ "${BUILD_MODE}" == "yocto" ]]; then
        EXTRA_ARGS+=(-b "${ARCH_BITS}")
        [[ -n "${CLI_SDK_ROOT}" ]]      && EXTRA_ARGS+=(-s "${CLI_SDK_ROOT}")
        [[ -n "${CLI_TOOLCHAIN_FILE}" ]] && EXTRA_ARGS+=(-t "${CLI_TOOLCHAIN_FILE}")
    else
        EXTRA_ARGS+=(-a "${TARGET_ARCH}")
    fi

    if bash "${BUILD_SCRIPT}" "${EXTRA_ARGS[@]}"; then
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

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped (${#SKIPPED[@]}):"
    for E in "${SKIPPED[@]}"; do echo "  - $E"; done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed (${#FAILED[@]}):"
    for E in "${FAILED[@]}"; do echo "  - $E"; done
    exit 1
fi

echo ""
echo "All examples built successfully!"

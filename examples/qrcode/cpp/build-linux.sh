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
    echo "Usage: $0 [-m <mode>] [-a <target_arch>] [-b <arch_bits>] [-s <yocto_sdk_root>] [-t <toolchain_file>]"
    echo "  -m <mode>      : Build mode: 'linux' or 'yocto' (default: linux)"
    echo "  -a <target>    : Target arch for linux mode (default: aarch64)"
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
while getopts 'm:a:b:s:t:h' opt; do
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

ROOT_PWD=$(cd "$(dirname $0)" && pwd)

# ===========================================================================
# Yocto build
# ===========================================================================
if [[ "${BUILD_MODE}" == "yocto" ]]; then

    if [[ "${ARCH_BITS}" != "32" && "${ARCH_BITS}" != "64" ]]; then
        echo "Unsupported ARCH_BITS \"${ARCH_BITS}\". Must be 32 or 64." >&2
        exit 1
    fi

    # Configurable via environment variables (CLI args > env vars > defaults)
    CMAKE_BIN="${CMAKE_BIN:-cmake}"
    YOCTO_SDK_ROOT="${CLI_SDK_ROOT:-${YOCTO_SDK_ROOT:-/mnt/fileroot/yuxuan.hu/my_poky_sdk}}"
    TOOLCHAIN_FILE="${CLI_TOOLCHAIN_FILE:-${TOOLCHAIN_FILE:-${ROOT_PWD}/../../cmake/yocto-toolchain.cmake}}"

    # Export variables for CMake
    export YOCTO_SDK_ROOT
    export ARCH_BITS

    BUILD_DIR="${ROOT_PWD}/build/yocto/${ARCH_BITS}"

    echo "==> Building Yocto ${ARCH_BITS}-bit"
    echo "    toolchain : ${TOOLCHAIN_FILE}"
    echo "    SDK root  : ${YOCTO_SDK_ROOT}"
    echo "    BUILD_DIR : ${BUILD_DIR}"

    mkdir -p "${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"

    # Select OpenCV based on target architecture
    if [[ "${ARCH_BITS}" == "32" ]]; then
        OPENCV_DIR="${ROOT_PWD}/../../../dependency/opencv/opencv-linux-armhf-yocto/share/OpenCV"
    else
        OPENCV_DIR="${ROOT_PWD}/../../../dependency/opencv/opencv-linux-aarch64-yocto/share/OpenCV"
    fi

    "${CMAKE_BIN}" \
        -S "${ROOT_PWD}/src" \
        -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DYOCTO_SDK_ROOT="${YOCTO_SDK_ROOT}" \
        -DARCH_BITS="${ARCH_BITS}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpenCV_DIR="${OPENCV_DIR}"

    "${CMAKE_BIN}" --build "${BUILD_DIR}" --config Release

    # Strip (best-effort)
    HOST_SYSROOT="${YOCTO_SDK_ROOT}/sysroots/x86_64-pokysdk-linux"
    if [[ "${ARCH_BITS}" == "32" ]]; then
        CROSS_TRIPLE="arm-poky-linux-gnueabi"
    else
        CROSS_TRIPLE="aarch64-poky-linux"
    fi
    STRIP_TOOL="${HOST_SYSROOT}/usr/bin/${CROSS_TRIPLE}/${CROSS_TRIPLE}-strip"
    if [[ -x "${STRIP_TOOL}" ]]; then
        "${STRIP_TOOL}" --strip-unneeded "${BUILD_DIR}/qr_demo"
    else
        echo "warning: strip tool not found; keeping debug info." >&2
    fi

    echo "Build complete. Executable in ${BUILD_DIR}/qr_demo"
    exit 0
fi

# ===========================================================================
# Standard Linux cross-compile build
# ===========================================================================

if [[ "${ARCH_BITS}" != "32" && "${ARCH_BITS}" != "64" ]]; then
    echo "Unsupported ARCH_BITS \"${ARCH_BITS}\". Must be 32 or 64." >&2
    exit 1
fi

if [[ "${ARCH_BITS}" == "32" ]]; then
    DEFAULT_COMPILER="arm-none-linux-gnueabihf"
    # Correct to the right arch
    if [[ "${TARGET_ARCH}" == "aarch64" ]]; then
        TARGET_ARCH="arm"
    fi
else
    DEFAULT_COMPILER="aarch64-none-linux-gnu"
fi

# Allow override via GCC_COMPILER environment variable
GCC_COMPILER=${GCC_COMPILER:-${DEFAULT_COMPILER}}

# Set compilers and strip tool
if [[ "${GCC_COMPILER}" == *"-gcc" ]]; then
    export CC="${GCC_COMPILER}"
    export CXX="${GCC_COMPILER%-gcc}-g++"
    STRIP_TOOL="${GCC_COMPILER%-gcc}-strip"
else
    export CC="${GCC_COMPILER}-gcc"
    export CXX="${GCC_COMPILER}-g++"
    STRIP_TOOL="${GCC_COMPILER}-strip"
fi

# Validate compiler is in PATH
if ! command -v "${CC}" &> /dev/null; then
    echo "Error: Compiler '${CC}' not found."
    echo "Please add your toolchain 'bin' directory to your PATH, or set GCC_COMPILER."
    echo "Example: export PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin:\$PATH"
    exit 1
fi

# Set up directories
BUILD_DIR="${ROOT_PWD}/build/linux/${ARCH_BITS}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"

echo "==> Building Linux ${ARCH_BITS}-bit"
echo "    COMPILER    : ${CC}"
echo "    TARGET_ARCH : ${TARGET_ARCH}"
echo "    BUILD_DIR   : ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"

if [[ "${ARCH_BITS}" == "32" ]]; then
    OPENCV_DIR="${ROOT_PWD}/../../../dependency/opencv/opencv-linux-armhf-buildroot/share/OpenCV"
else
    OPENCV_DIR="${ROOT_PWD}/../../../dependency/opencv/opencv-linux-aarch64-buildroot/share/OpenCV"
fi

"${CMAKE_BIN}" \
    -S "${ROOT_PWD}/src" \
    -B "${BUILD_DIR}" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR="${TARGET_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR="${OPENCV_DIR}" \
    -DARCH_BITS="${ARCH_BITS}" \
    -DCMAKE_CXX_FLAGS="-Wno-psabi"

# Build
"${CMAKE_BIN}" --build "${BUILD_DIR}" --config Release -j4

# Strip binary
if command -v "${STRIP_TOOL}" &> /dev/null; then
    "${STRIP_TOOL}" --strip-unneeded "${BUILD_DIR}/qr_demo"
else
    echo "warning: strip tool '${STRIP_TOOL}' not found; keeping debug info." >&2
fi

echo "Build complete. Executable in ${BUILD_DIR}/qr_demo"
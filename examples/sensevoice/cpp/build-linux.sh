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
    echo "Usage: $0 [-m <mode>] [-a <target_arch>] [-b <arch_bits>] [-s <yocto_sdk_root>] [-t <toolchain_file>]"
    echo "  -m <mode>      : Build mode (default: linux)"
    echo "                    Supported values:"
    echo "                      linux"
    echo "                      yocto"
    echo "  -a <target>    : Target arch for linux mode (default: aarch64)"
    echo "                    Supported values:"
    echo "                      aarch64"
    echo "                      armhf"
    echo "  -b <arch_bits> : Arch bits for yocto mode (default: 64)"
    echo "                    Supported values:"
    echo "                      32"
    echo "                      64"
    echo "  -s <sdk_root>  : Yocto SDK root path (overrides YOCTO_SDK_ROOT env var)"
    echo "  -t <toolchain> : CMake toolchain file (overrides TOOLCHAIN_FILE env var)"
    echo "  -h             : Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  AMLNN_HOME"
    echo "  GCC_COMPILER   (linux mode, default: aarch64-linux-gnu)"
    echo "  YOCTO_SDK_ROOT (yocto mode)"
    exit 1
}

BUILD_MODE=linux
TARGET_ARCH=aarch64
ARCH_BITS=64
CLI_SDK_ROOT=""
CLI_TOOLCHAIN_FILE=""

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

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ===========================================================================
# Yocto build
# ===========================================================================
if [[ "${BUILD_MODE}" == "yocto" ]]; then

    if [[ "${ARCH_BITS}" != "32" && "${ARCH_BITS}" != "64" ]]; then
        echo "Unsupported ARCH_BITS \"${ARCH_BITS}\". Must be 32 or 64." >&2
        exit 1
    fi

    CMAKE_BIN="${CMAKE_BIN:-cmake}"
    YOCTO_SDK_ROOT="${CLI_SDK_ROOT:-${YOCTO_SDK_ROOT:-/data/yuandian/tools/poky/4.0.20}}"
    TOOLCHAIN_FILE="${CLI_TOOLCHAIN_FILE:-${TOOLCHAIN_FILE:-${ROOT_PWD}/../../cmake/yocto-toolchain.cmake}}"

    export YOCTO_SDK_ROOT
    export ARCH_BITS

    BUILD_DIR="${ROOT_PWD}/build/yocto/${ARCH_BITS}"

    echo "==> Building sensevoice for Yocto ${ARCH_BITS}-bit"
    echo "    toolchain : ${TOOLCHAIN_FILE}"
    echo "    SDK root  : ${YOCTO_SDK_ROOT}"
    echo "    BUILD_DIR : ${BUILD_DIR}"

    mkdir -p "${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    "${CMAKE_BIN}" \
        -S "${ROOT_PWD}/src" \
        -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DYOCTO_SDK_ROOT="${YOCTO_SDK_ROOT}" \
        -DARCH_BITS="${ARCH_BITS}" \
        -DAMLNN_HOME=${AMLNN_HOME:-} \
        -DCMAKE_BUILD_TYPE=Release

    "${CMAKE_BIN}" --build "${BUILD_DIR}" --config Release -j"${JOBS}"

    HOST_SYSROOT="${YOCTO_SDK_ROOT}/sysroots/x86_64-pokysdk-linux"
    if [[ "${ARCH_BITS}" == "32" ]]; then
        CROSS_TRIPLE="arm-poky-linux-gnueabi"
    else
        CROSS_TRIPLE="aarch64-poky-linux"
    fi
    STRIP_TOOL="${HOST_SYSROOT}/usr/bin/${CROSS_TRIPLE}/${CROSS_TRIPLE}-strip"
    if [[ -x "${STRIP_TOOL}" ]]; then
        "${STRIP_TOOL}" --strip-unneeded "${BUILD_DIR}/sensevoice_demo"
    else
        echo "warning: strip tool not found; keeping debug info." >&2
    fi

    echo "Build complete."
    echo "  Static library: ${BUILD_DIR}/libsensevoice.a"
    echo "  Demo binary:    ${BUILD_DIR}/sensevoice_demo"
    exit 0
fi

if [[ "${BUILD_MODE}" != "linux" ]]; then
    echo "Unsupported build mode \"${BUILD_MODE}\". Must be linux or yocto." >&2
    exit 1
fi

if [[ "${TARGET_ARCH}" != "aarch64" && "${TARGET_ARCH}" != "armhf" ]]; then
    echo "Unsupported TARGET_ARCH \"${TARGET_ARCH}\". Must be aarch64 or armhf." >&2
    exit 1
fi

# ===========================================================================
# Standard Linux cross-compile build
# ===========================================================================

if [[ "${TARGET_ARCH}" == "armhf" ]]; then
    LINUX_ARCH_BITS=32
    GCC_COMPILER=${GCC_COMPILER:-arm-linux-gnueabihf}
else
    LINUX_ARCH_BITS=64
    GCC_COMPILER=${GCC_COMPILER:-aarch64-linux-gnu}
fi

if [[ ${GCC_COMPILER} == *"-gcc" ]]; then
    export CC=${GCC_COMPILER}
    export CXX=${GCC_COMPILER%-gcc}-g++
else
    export CC=${GCC_COMPILER}-gcc
    export CXX=${GCC_COMPILER}-g++
fi

if ! command -v "${CC}" &> /dev/null; then
    echo "Error: Compiler ${CC} not found."
    echo "Please set GCC_COMPILER to your cross-compiler path prefix."
    echo "Example: export GCC_COMPILER=/path/to/toolchain/bin/aarch64-linux-gnu"
    exit 1
fi

BUILD_DIR="${ROOT_PWD}/build/linux/${TARGET_ARCH}"

echo "Building sensevoice for Linux..."
echo "COMPILER:    ${CC}"
echo "TARGET_ARCH: ${TARGET_ARCH}"
echo "BUILD_DIR:   ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"

cmake -Wno-dev \
    -S "${ROOT_PWD}/src" \
    -B "${BUILD_DIR}" \
    -DAMLNN_HOME=${AMLNN_HOME:-} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR="${TARGET_ARCH}" \
    -DARCH_BITS="${LINUX_ARCH_BITS}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo "Build complete."
echo "  Static library: ${BUILD_DIR}/libsensevoice.a"
echo "  Demo binary:    ${BUILD_DIR}/sensevoice_demo"

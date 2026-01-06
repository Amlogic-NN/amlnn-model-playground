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
    echo "Usage: $0 [-a <target_arch>]"
    echo "  -a <target_arch> : Target architecture (default: aarch64)"
    echo "  -h               : Show this help message"
    exit 1
}

# Default values
TARGET_ARCH=aarch64

# Parse arguments
while getopts 'a:h' opt; do
  case "$opt" in
    a)
      TARGET_ARCH=$OPTARG
      ;;
    h)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

# Default to aarch64-linux-gnu if GCC_COMPILER is not set
GCC_COMPILER=${GCC_COMPILER:-aarch64-linux-gnu}

# Set compilers
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

# Validate compiler
if ! command -v ${CC} &> /dev/null; then
    echo "Error: Compiler ${CC} not found."
    echo "Please set GCC_COMPILER environment variable to your cross-compiler path prefix."
    echo "Example: export GCC_COMPILER=/path/to/toolchain/bin/aarch64-linux-gnu"
    exit 1
fi

ROOT_PWD=$(cd "$(dirname $0)" && pwd)
BUILD_DIR=${ROOT_PWD}/build/linux

echo "Building for Linux..."
echo "COMPILER: ${CC}"
echo "TARGET_ARCH: ${TARGET_ARCH}"
echo "BUILD_DIR: ${BUILD_DIR}"

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ../../src \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_BUILD_TYPE=Release

make -j4

echo "Build complete. Executable in ${BUILD_DIR}/yolov8_demo"

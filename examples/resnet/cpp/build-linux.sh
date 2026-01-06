#TODO
#!/bin/bash
set -e

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

echo "Build complete. Executable in ${BUILD_DIR}/resnet_demo"

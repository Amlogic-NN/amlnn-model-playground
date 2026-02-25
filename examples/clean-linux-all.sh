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
    echo "Usage: $0 [-m <mode>]"
    echo "  -m <mode> : What to clean: 'linux', 'yocto', or 'all' (default: all)"
    echo "  -h        : Show this help message"
    exit 1
}

# Default values
CLEAN_MODE=all

# Parse arguments
while getopts 'm:h' opt; do
  case "$opt" in
    m)
      CLEAN_MODE=$OPTARG
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

case "${CLEAN_MODE}" in
  linux)
    echo "Cleaning all Linux build directories..."
    find "${SCRIPT_DIR}" -type d -path "*/build/linux" | while read dir; do
        echo "  rm -rf ${dir}"
        rm -rf "${dir}"
    done
    ;;
  yocto)
    echo "Cleaning all Yocto build directories..."
    find "${SCRIPT_DIR}" -type d -path "*/build/yocto" | while read dir; do
        echo "  rm -rf ${dir}"
        rm -rf "${dir}"
    done
    ;;
  all)
    echo "Cleaning all Linux and Yocto build directories..."
    find "${SCRIPT_DIR}" -type d \( -path "*/build/linux" -o -path "*/build/yocto" \) | while read dir; do
        echo "  rm -rf ${dir}"
        rm -rf "${dir}"
    done
    ;;
  *)
    echo "Unknown mode: ${CLEAN_MODE}" >&2
    usage
    ;;
esac

echo "Done."

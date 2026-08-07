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

set -euo pipefail

usage() {
    echo "Usage: $0 [-m <mode>] [-b <arch_bits>]"
    echo "  -m <mode>      : What to clean: linux, yocto, or all (default: all)"
    echo "  -b <arch_bits> : Architecture to clean: 32, 64, or all (default: all)"
    echo "  -h             : Show this help message"
    exit 1
}

CLEAN_MODE=all
ARCH_BITS=all

while getopts 'm:b:h' opt; do
    case "$opt" in
        m) CLEAN_MODE=$OPTARG ;;
        b) ARCH_BITS=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ "${CLEAN_MODE}" != "linux" && "${CLEAN_MODE}" != "yocto" && "${CLEAN_MODE}" != "all" ]]; then
    echo "Error: Mode must be linux, yocto, or all."
    exit 1
fi

if [[ "${ARCH_BITS}" != "32" && "${ARCH_BITS}" != "64" && "${ARCH_BITS}" != "all" ]]; then
    echo "Error: Arch bits must be 32, 64, or all."
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

MODES=()
BITS=()

if [[ "${CLEAN_MODE}" == "all" ]]; then
    MODES=("linux" "yocto")
else
    MODES=("${CLEAN_MODE}")
fi

if [[ "${ARCH_BITS}" == "all" ]]; then
    BITS=("32" "64")
else
    BITS=("${ARCH_BITS}")
fi

for mode in "${MODES[@]}"; do
    for bits in "${BITS[@]}"; do
        echo "Cleaning ${mode} ${bits}-bit build directories..."

        while IFS= read -r -d '' dir; do
            echo "  rm -rf ${dir}"
            rm -rf "${dir}"
        done < <(find "${SCRIPT_DIR}" -type d -path "*/build/${mode}/${bits}" -print0)
    done
done

echo "Done."

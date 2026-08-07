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
    echo "Usage: $0 [-a <target_abi>]"
    echo "  -a <target_abi> : ABI to clean:"
    echo "                    arm64-v8a"
    echo "                    armeabi-v7a"
    echo "                    all (default)"
    echo "  -h              : Show this help message"
    exit 1
}

TARGET_ABI=all

while getopts 'a:h' opt; do
    case "$opt" in
        a) TARGET_ABI=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ "${TARGET_ABI}" != "arm64-v8a" && "${TARGET_ABI}" != "armeabi-v7a" && "${TARGET_ABI}" != "all" ]]; then
    echo "Error: ABI must be arm64-v8a, armeabi-v7a, or all."
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

ABIS=()
if [[ "${TARGET_ABI}" == "all" ]]; then
    ABIS=("arm64-v8a" "armeabi-v7a")
else
    ABIS=("${TARGET_ABI}")
fi

for abi in "${ABIS[@]}"; do
    echo "Cleaning Android ${abi} build directories..."

    while IFS= read -r -d '' dir; do
        echo "  rm -rf ${dir}"
        rm -rf "${dir}"
    done < <(find "${SCRIPT_DIR}" -type d -path "*/build/android/${abi}" -print0)
done

echo "Done."
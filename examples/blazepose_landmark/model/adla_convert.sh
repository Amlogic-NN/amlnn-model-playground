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

# 1. $1: set ADLA_TOOL_PATH 
# 2. $2: set  target-platform 
#  for A311D2 target-platform is PRODUCT_PID0XA003
#  for S905X5 target-platform is PRODUCT_PID0XA005
# Usage: ./adla_convert.sh pose_landmark_full.tflite /XXX/adla-toolkit-binary-3.2.9.3 PRODUCT_PID0XA005

model_path=$1
ADLA_TOOL_PATH=$2
target_platform=$3

echo "model_path:[$model_path]"
echo "ADLA_TOOL_PATH:[$ADLA_TOOL_PATH]"
echo "target-platform:[$target_platform]"

adla_convert=${ADLA_TOOL_PATH}/bin/adla_convert

$adla_convert --model-type tflite \
	--model $model_path \
	--inputs input_1 --input-shapes "256,256,3" \
	--quantize-dtype int16 \
	--source-file dataset_coco.txt \
	--channel-mean-value "0,0,0,255" \
	--target-platform $target_platform \
	--disable-per-channel false

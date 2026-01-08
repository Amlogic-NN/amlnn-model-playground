# 1. $1: set ADLA_TOOL_PATH 
# 2. $2: set  target-plaftorm 
#  for A311D2 target-platform is PRODUCT_PID0XA003
#  for S905X5 target-platform is PRODUCT_PID0XA005
# Usage: ./adla_covnert.sh yolov8m.onnx /XXX/adla-toolkit-binary-3.2.9.3 PRODUCT_PID0XA005

model_path=$1
ADLA_TOOL_PATH=$2
target_platform=$3

echo "model_path:[$model_path]"
echo "ADLA_TOOL_PATH:[$ADLA_TOOL_PATH]"
echo "target-plaftorm:[$target_platform]"

adla_convert=${ADLA_TOOL_PATH}/bin/adla_convert

$adla_convert --model-type onnx \
	--model $model_path \
	--inputs images --input-shapes "1,3,640,640" \
	--quantize-dtype int8 \
	--source-file dataset_coco.txt \
	--channel-mean-value "0,0,0,255" \
	--outputs "/model.22/Concat_output_0 /model.22/Concat_1_output_0 /model.22/Concat_2_output_0" \
	--target-platform $target_platform

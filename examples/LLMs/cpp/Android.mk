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

LOCAL_PATH := $(call my-dir)
LLM_SDK_PATH := $(LOCAL_PATH)/../../../../amlnn-toolkit/nn_runtime/llmsdk
3RDPARTY_PATH := $(LOCAL_PATH)/../../../dependency
$(warning $(LOCAL_PATH))

include $(CLEAR_VARS)

LOCAL_SRC_FILES := main.cpp

LOCAL_C_INCLUDES := \
	$(LLM_SDK_PATH)/include \

LOCAL_LDFLAGS := \
	-L$(LLM_SDK_PATH)/android/arm64-v8a -lllmsdk

LOCAL_LDLIBS := -llog -ldl -lm -fuse-ld=ld

LOCAL_MODULE := demo_llm_main

LOCAL_LICENSE_KINDS := SPDX-License-Identifier: Apache-2.0

LOCAL_LICENSE_CONDITIONS := notice

LOCAL_LICENSE_COMMENTS := Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.

LOCAL_LICENSE_URL := https://www.apache.org/licenses/LICENSE-2.0

LOCAL_LICENSE_FILE := LICENSE


include $(BUILD_EXECUTABLE)

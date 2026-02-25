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

include $(BUILD_EXECUTABLE)

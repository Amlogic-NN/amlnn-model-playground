LOCAL_PATH := $(call my-dir)
LLM_SDK_PATH := $(LOCAL_PATH)/../../01_src
3RDPARTY_PATH := $(LOCAL_PATH)/../../3rdparty
$(warning $(LOCAL_PATH))

include $(CLEAR_VARS)

LOCAL_SRC_FILES := main.cpp

LOCAL_C_INCLUDES := \
	$(LLM_SDK_PATH)/jni \
	$(3RDPARTY_PATH)/include \

LOCAL_LDFLAGS := \
	-L$(LLM_SDK_PATH)/libs/arm64-v8a -lllmsdk

LOCAL_LDLIBS := -llog -ldl -lm -fuse-ld=ld

LOCAL_MODULE := demo_llm_main

include $(BUILD_EXECUTABLE)

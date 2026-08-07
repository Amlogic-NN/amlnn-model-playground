APP_PLATFORM := android-28
APP_CPPFLAGS += -std=c++17
APP_STL      := c++_static

APP_CFLAGS   := -Wno-error=format-security

# APP_ABI      := armeabi-v7a
APP_ABI      := arm64-v8a
#APP_ABI      := armeabi armeabi-v7a arm64-v8a

APP_OPTION   := release

# APP_BUILD_SCRIPT := Android.mk
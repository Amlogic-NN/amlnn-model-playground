#!/bin/bash
if [ ! -d "$NDK_PATH" ]; then
    echo "Error: NDK_PATH '$NDK_PATH' not found."
    echo "Please set NDK_PATH environment variable to your Android NDK directory."
    exit 1
fi

$NDK_PATH/ndk-build \
    NDK_PROJECT_PATH=. \
    APP_BUILD_SCRIPT=./Android.mk \
    NDK_APPLICATION_MK=./Application.mk

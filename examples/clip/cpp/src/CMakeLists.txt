cmake_minimum_required(VERSION 3.5)
project(clip_demo)

set(CMAKE_CXX_STANDARD 17)

# Set NNSDK path
set(NNSDK_ROOT "${CMAKE_SOURCE_DIR}/../../../../dependency/nnsdk")
include_directories(${NNSDK_ROOT}/include)
include_directories(${CMAKE_SOURCE_DIR}/../../../../common)

# Set 3rdparty path
set(3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/../../../../dependency")

# Include directories for stb_image and json
# Note: code uses #include "stb_image.h" and #include "json.hpp"
include_directories(${3RDPARTY_DIR}/stb_image)
include_directories(${3RDPARTY_DIR}/json)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        link_directories(${NNSDK_ROOT}/lib/android/arm64-v8a)
    else()
        link_directories(${NNSDK_ROOT}/lib/android/armeabi-v7a)
    endif()
    # Android needs log
    link_libraries(log)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${NNSDK_ROOT}/lib/linux/lib64_yocto)
endif()

add_executable(${PROJECT_NAME}
    main.cpp
    model_invoke.cpp
    pre_postprocess.cpp
)

target_link_libraries(${PROJECT_NAME}
    nnsdk
    dl
    m
)


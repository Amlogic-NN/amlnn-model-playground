cmake_minimum_required(VERSION 3.5)
project(retinaface_demo) 

set(CMAKE_CXX_STANDARD 17)

# Set NNSDK path
if(NOT DEFINED NNSDK_DIR)
    set(NNSDK_DIR "${CMAKE_SOURCE_DIR}/../../../../../amlnn-toolkit/nn_runtime/nnsdk")
endif()
set(NNSDK_ROOT "${NNSDK_DIR}")
message(STATUS "NNSDK_ROOT: ${NNSDK_ROOT}")

include_directories(${NNSDK_ROOT}/include)
include_directories(${CMAKE_SOURCE_DIR}/../../../../common)

# Set dependency path
set(3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/../../../../dependency")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        link_directories(${NNSDK_ROOT}/android/arm64-v8a)
    else()
        link_directories(${NNSDK_ROOT}/android/armeabi-v7a)
    endif()
    # Android needs log
    link_libraries(log)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    link_directories(${NNSDK_ROOT}/linux/yocto/aarch64-poky-linux)
endif()

# Find OpenCV
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(retinaface_demo
    main.cpp
    postprocess.cpp
    ${CMAKE_SOURCE_DIR}/../../../../common/model_loader.cpp 
)

target_link_libraries(retinaface_demo
    ${OpenCV_LIBS}
    nnsdk
)
cmake_minimum_required(VERSION 3.5)
project(yolo_world_demo)

set(CMAKE_CXX_STANDARD 17)

# Set NNSDK path
# Set NNSDK path
set(NNSDK_ROOT "${CMAKE_SOURCE_DIR}/../../../../dependency/nnsdk")
include_directories(${NNSDK_ROOT}/include)
include_directories(${CMAKE_SOURCE_DIR}/../../../../common)

# Set 3rdparty path
set(3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/../../../../dependency")

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

# Find OpenCV
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(paddleocr_det_demo
    main.cpp
    postprocess.cpp
    postprocess.h
    clipper.cpp
    clipper.h
    ${CMAKE_SOURCE_DIR}/../../../../common/model_loader.cpp
)

target_link_libraries(paddleocr_det_demo
    ${OpenCV_LIBS}
    nnsdk
)

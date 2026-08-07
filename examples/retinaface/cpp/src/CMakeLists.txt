cmake_minimum_required(VERSION 3.10...3.27)
project(retinaface_demo)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../../cmake")
find_package(AMLNN REQUIRED)
include_directories(${AMLNN_INCLUDE_DIR})
link_directories(${AMLNN_LIBRARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/../../../../common)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android needs log
    link_libraries(log)
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
    ${AMLNN_LIBRARY}
)
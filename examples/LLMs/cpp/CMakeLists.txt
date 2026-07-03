cmake_minimum_required(VERSION 3.10...3.27)
project(LLM_demo)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")

# Find LLMSDK
find_package(AMLLLM REQUIRED)
include_directories(${AMLNN_LLM_INCLUDE_DIR})
link_directories(${AMLNN_LLM_LIBRARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/../../../dependency/json)

# Set 3rdparty path
set(3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/../../../../dependency")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android needs log
    link_libraries(log)
endif()

add_executable(LLM_demo
    main.cpp
)

target_link_libraries(LLM_demo
    ${AMLNN_LLM_LIBRARY}
)
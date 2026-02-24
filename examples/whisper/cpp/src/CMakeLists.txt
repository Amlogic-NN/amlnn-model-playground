cmake_minimum_required(VERSION 3.10...3.27)
project(whisper_demo)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../../cmake")
find_package(AMLNN REQUIRED)
include_directories(${AMLNN_INCLUDE_DIR})
link_directories(${AMLNN_LIBRARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/../../../../common)

# Set 3rdparty path
set(3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/../../../../dependency")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android needs log
    link_libraries(log)
endif()

add_executable(${PROJECT_NAME}
    main.cpp
    common.cpp
    whisper.cpp
    whisper_invoke.cpp
    pre_process_whisper.cpp
    post_process_whisper.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${AMLNN_LIBRARY}
    dl
    m
)


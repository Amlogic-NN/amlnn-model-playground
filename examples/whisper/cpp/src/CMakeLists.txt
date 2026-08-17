cmake_minimum_required(VERSION 3.10...3.27)
project(whisper_demo)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../../cmake")
find_package(AMLNN REQUIRED)
include_directories(${AMLNN_INCLUDE_DIR})
link_directories(${AMLNN_LIBRARY_DIR})

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android needs log
    link_libraries(log)
endif()

find_package(Threads REQUIRED)

add_executable(whisper_demo
    main.cpp
    common.cpp
    whisper.cpp
    whisper_invoke.cpp
    pre_process_whisper.cpp
    post_process_whisper.cpp
)

target_link_libraries(whisper_demo
    ${AMLNN_LIBRARY}
    Threads::Threads
)



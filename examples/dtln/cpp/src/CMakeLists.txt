cmake_minimum_required(VERSION 3.10...3.27)
project(dtln_demo)

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

add_library(kissfft STATIC
    kissfft/kiss_fft.c
    kissfft/kiss_fftr.c
)

target_include_directories(kissfft PUBLIC
    kissfft
)

add_executable(dtln_demo
    main.cpp
    ${CMAKE_SOURCE_DIR}/../../../../common/model_loader.cpp
)

target_link_libraries(dtln_demo
    kissfft
)

target_link_libraries(dtln_demo
    kissfft
    ${AMLNN_LIBRARY}
)

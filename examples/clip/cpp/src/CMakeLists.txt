cmake_minimum_required(VERSION 3.10...3.27)
project(clip_demo)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../../cmake")
find_package(AMLNN REQUIRED)
include_directories(${AMLNN_INCLUDE_DIR})
link_directories(${AMLNN_LIBRARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/../../../../common)

# Set 3rdparty path
set(3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/../../../../dependency")

# Include directories for stb_image and json
# Note: code uses #include "stb_image.h" and #include "json.hpp"
include_directories(${3RDPARTY_DIR}/stb_image)
include_directories(${3RDPARTY_DIR}/json)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android needs log
    link_libraries(log)
endif()

add_executable(${PROJECT_NAME}
    main.cpp
    model_invoke.cpp
    postprocess.cpp
    clip_tokenizer.cpp
    ${CMAKE_SOURCE_DIR}/../../../../common/model_loader.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${AMLNN_LIBRARY}
)


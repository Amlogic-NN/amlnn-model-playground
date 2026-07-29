cmake_minimum_required(VERSION 3.10...3.27)
project(mobileclip_demo)

set(CMAKE_CXX_STANDARD 17)

# ============================================================
# Common 3rd party headers
# ============================================================
set(3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/../../../../dependency")
include_directories(${3RDPARTY_DIR}/stb_image)
include_directories(${3RDPARTY_DIR}/json)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    link_libraries(log)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../../cmake")
find_package(AMLNN REQUIRED)
include_directories(${AMLNN_INCLUDE_DIR})
link_directories(${AMLNN_LIBRARY_DIR})

add_executable(${PROJECT_NAME}
    main.cpp
    pre_postprocess.cpp
    clip_tokenizer.cpp
    model_invoke.cpp
    adla_mobileclip.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
        ${AMLNN_LIBRARY}
        ${CMAKE_DL_LIBS}
        log
        m
        z
    )
else()
    target_link_libraries(${PROJECT_NAME}
        ${AMLNN_LIBRARY}
        ${CMAKE_DL_LIBS}
        m
        pthread
    )
endif()

message(STATUS "MobileCLIP: Building ADLA-only runtime backend")

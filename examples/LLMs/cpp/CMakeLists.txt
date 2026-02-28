cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_SYSTEM_NAME Linux)
project(AML_LLM_NNSDK)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# SDK paths
set(LLM_NNSDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../01_src/jni")
set(NNSDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../01_src/jni/nnsdk_v2.8.1_2025_0317/lib/linux/lib64_yocto")

# Include header paths
include_directories(
    ${LLM_NNSDK_PATH}
)

# Source files
aux_source_directory(. SRC_LIST)

# Build executable
add_executable(demo_llm_yocto ${SRC_LIST})

# Link libraries
target_link_libraries(demo_llm_yocto
    ${LLM_NNSDK_PATH}/build/libllm_nnsdk.so
    ${NNSDK_PATH}/libnnsdk.so
    pthread
    m
    dl
)

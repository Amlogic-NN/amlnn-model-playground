cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_SYSTEM_NAME Linux)
project(AML_LLM_NNSDK)

# xinxin, when building the .so with Yocto using CMake, you can remove the sysroot settings
# from these CMakeLists.txt files and use the officially recommended approach instead:
# after sourcing the environment script, many environment variables will be set (check with `export`),
# and CMake will configure itself automatically based on them without needing explicit settings here.
# source /mnt/fileroot/xinxin.he/environment/new-yocto/64/environment-setup-armv8a-poky-linux
# export CXXFLAGS=$(echo "$CXXFLAGS" | sed 's/-g//g')
# export CFLAGS=$(echo "$CXXFLAGS" | sed 's/-g//g')
# cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE}  ..

# # Set Yocto cross-compilation environment
# set(SYSROOT_PATH /mnt/fileroot/xinxin.he/environment/new-yocto/64/sysroots/x86_64-pokysdk-linux)
# set(CMAKE_SYSROOT "${SYSROOT_PATH}")
# message(STATUS "Using sysroot path as ${SYSROOT_PATH}")

# include(CMakeForceCompiler)
# cmake_force_c_compiler("${SYSROOT_PATH}/usr/bin/aarch64-poky-linux/aarch64-poky-linux-gcc" GNU)
# cmake_force_cxx_compiler("${SYSROOT_PATH}/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++" GNU)

# # Set the sysroot for the actual target board
# set(MYSYSROOT "/mnt/fileroot/xinxin.he/environment/new-yocto/64/sysroots/armv8a-poky-linux")
# add_definitions("--sysroot=${MYSYSROOT}")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${MYSYSROOT}" CACHE INTERNAL "" FORCE)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${MYSYSROOT}" CACHE INTERNAL "" FORCE)
# set(CMAKE_FIND_ROOT_PATH "${MYSYSROOT}")
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

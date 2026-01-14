cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_SYSTEM_NAME Linux)
project(AML_LLM_NNSDK)

# xinxin, 后面yocto的编so的时候，如果用的cmake的话，可以把这些cmakelist.txt里面指定sysroot的都去掉，
# 然后用官方推荐的方式，source完之后它会增加很多环境变量，你可以export命令看一下，后续cmake根据环境变量自己会配置，就不用再cmakelist里面配置了
# source /mnt/fileroot/xinxin.he/environment/new-yocto/64/environment-setup-armv8a-poky-linux
# export CXXFLAGS=$(echo "$CXXFLAGS" | sed 's/-g//g')
# export CFLAGS=$(echo "$CXXFLAGS" | sed 's/-g//g')
# cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE}  ..

# # 设置 Yocto 交叉编译环境
# set(SYSROOT_PATH /mnt/fileroot/xinxin.he/environment/new-yocto/64/sysroots/x86_64-pokysdk-linux)
# set(CMAKE_SYSROOT "${SYSROOT_PATH}")
# message(STATUS "Using sysroot path as ${SYSROOT_PATH}")

# include(CMakeForceCompiler)
# cmake_force_c_compiler("${SYSROOT_PATH}/usr/bin/aarch64-poky-linux/aarch64-poky-linux-gcc" GNU)
# cmake_force_cxx_compiler("${SYSROOT_PATH}/usr/bin/aarch64-poky-linux/aarch64-poky-linux-g++" GNU)

# # 设置真正目标板的 sysroot
# set(MYSYSROOT "/mnt/fileroot/xinxin.he/environment/new-yocto/64/sysroots/armv8a-poky-linux")
# add_definitions("--sysroot=${MYSYSROOT}")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${MYSYSROOT}" CACHE INTERNAL "" FORCE)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${MYSYSROOT}" CACHE INTERNAL "" FORCE)
# set(CMAKE_FIND_ROOT_PATH "${MYSYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# SDK路径
set(LLM_NNSDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../01_src/jni")
set(NNSDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../01_src/jni/nnsdk_v2.8.1_2025_0317/lib/linux/lib64_yocto")

# 包含头文件路径
include_directories(
    ${LLM_NNSDK_PATH}
)

# 源文件
aux_source_directory(. SRC_LIST)

# 生成可执行文件
add_executable(demo_llm_yocto ${SRC_LIST})

# 链接库
target_link_libraries(demo_llm_yocto
    ${LLM_NNSDK_PATH}/build/libllm_nnsdk.so
    ${NNSDK_PATH}/libnnsdk.so
    pthread
    m
    dl
)

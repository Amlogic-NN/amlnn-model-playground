# Yocto cross-compilation toolchain
#
# Required cache variables (pass via -D):
#   YOCTO_SDK_ROOT  - Yocto SDK root dir (contains sysroots/)
#   ARCH_BITS       - 32 or 64
#
# Example:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/yocto-toolchain.cmake \
#         -DYOCTO_SDK_ROOT=/opt/poky/4.0.20 \
#         -DARCH_BITS=64 ...

set(CMAKE_SYSTEM_NAME Linux)

# Fallback to environment variables if not defined via -D
if(NOT DEFINED YOCTO_SDK_ROOT AND DEFINED ENV{YOCTO_SDK_ROOT})
    set(YOCTO_SDK_ROOT "$ENV{YOCTO_SDK_ROOT}")
endif()

if(NOT DEFINED ARCH_BITS AND DEFINED ENV{ARCH_BITS})
    set(ARCH_BITS "$ENV{ARCH_BITS}")
endif()

# Mandatory check
if(NOT DEFINED YOCTO_SDK_ROOT)
    message(FATAL_ERROR "YOCTO_SDK_ROOT must point to the Yocto SDK root directory")
endif()

if(NOT DEFINED ARCH_BITS)
    message(FATAL_ERROR "ARCH_BITS must be set to 32 or 64")
endif()

# Propagate these variables to try_compile sub-projects
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES YOCTO_SDK_ROOT ARCH_BITS)

set(_HOST_SYSROOT "${YOCTO_SDK_ROOT}/sysroots/x86_64-pokysdk-linux")
if(NOT IS_DIRECTORY "${_HOST_SYSROOT}")
    message(FATAL_ERROR "Yocto host sysroot not found: ${_HOST_SYSROOT}")
endif()

if(ARCH_BITS EQUAL 32)
    set(_TARGET_SYSROOT "${YOCTO_SDK_ROOT}/sysroots/armv7at2hf-neon-poky-linux-gnueabi")
    set(_TRIPLE        "arm-poky-linux-gnueabi")
    set(_ARCH_FLAGS    "-march=armv7-a -marm -mfpu=vfp -mfloat-abi=hard")
elseif(ARCH_BITS EQUAL 64)
    set(_TARGET_SYSROOT "${YOCTO_SDK_ROOT}/sysroots/armv8a-poky-linux")
    set(_TRIPLE        "aarch64-poky-linux")
    set(_ARCH_FLAGS    "")
else()
    message(FATAL_ERROR "Unsupported ARCH_BITS: ${ARCH_BITS} (expected 32 or 64)")
endif()

if(NOT IS_DIRECTORY "${_TARGET_SYSROOT}")
    message(FATAL_ERROR "Yocto target sysroot not found: ${_TARGET_SYSROOT}")
endif()

set(CMAKE_C_COMPILER   "${_HOST_SYSROOT}/usr/bin/${_TRIPLE}/${_TRIPLE}-gcc")
set(CMAKE_CXX_COMPILER "${_HOST_SYSROOT}/usr/bin/${_TRIPLE}/${_TRIPLE}-g++")

set(CMAKE_SYSROOT               "${_TARGET_SYSROOT}")
set(CMAKE_C_FLAGS_INIT          "--sysroot=${_TARGET_SYSROOT} ${_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT        "--sysroot=${_TARGET_SYSROOT} ${_ARCH_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "--sysroot=${_TARGET_SYSROOT}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "--sysroot=${_TARGET_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH              "${_TARGET_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Expose ARCH_BITS and triple for CMakeLists.txt to use
set(YOCTO_CROSS_TRIPLE "${_TRIPLE}" CACHE INTERNAL "")

# Yocto cross-compilation toolchain
#
# Required cache variables:
#   YOCTO_SDK_ROOT - Yocto SDK root containing sysroots/
#   ARCH_BITS      - 32 or 64

set(CMAKE_SYSTEM_NAME Linux)

if(NOT DEFINED YOCTO_SDK_ROOT AND DEFINED ENV{YOCTO_SDK_ROOT})
    set(YOCTO_SDK_ROOT "$ENV{YOCTO_SDK_ROOT}")
endif()

if(NOT DEFINED ARCH_BITS AND DEFINED ENV{ARCH_BITS})
    set(ARCH_BITS "$ENV{ARCH_BITS}")
endif()

if(NOT DEFINED YOCTO_SDK_ROOT)
    message(FATAL_ERROR "YOCTO_SDK_ROOT must point to the Yocto SDK root directory")
endif()

if(NOT DEFINED ARCH_BITS)
    message(FATAL_ERROR "ARCH_BITS must be set to 32 or 64")
endif()

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES YOCTO_SDK_ROOT ARCH_BITS)

set(_HOST_SYSROOT "${YOCTO_SDK_ROOT}/sysroots/x86_64-pokysdk-linux")
if(NOT IS_DIRECTORY "${_HOST_SYSROOT}")
    message(FATAL_ERROR "Yocto host sysroot not found: ${_HOST_SYSROOT}")
endif()

if(ARCH_BITS EQUAL 32)
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(_TARGET_SYSROOT "${YOCTO_SDK_ROOT}/sysroots/armv7at2hf-neon-poky-linux-gnueabi")
    set(_TRIPLE "arm-poky-linux-gnueabi")
    set(_ARCH_FLAGS "-march=armv7-a -marm -mfpu=neon -mfloat-abi=hard")
elseif(ARCH_BITS EQUAL 64)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(_TARGET_SYSROOT "${YOCTO_SDK_ROOT}/sysroots/armv8a-poky-linux")
    set(_TRIPLE "aarch64-poky-linux")
    set(_ARCH_FLAGS "")
else()
    message(FATAL_ERROR "Unsupported ARCH_BITS: ${ARCH_BITS} (expected 32 or 64)")
endif()

if(NOT IS_DIRECTORY "${_TARGET_SYSROOT}")
    message(FATAL_ERROR "Yocto target sysroot not found: ${_TARGET_SYSROOT}")
endif()

set(CMAKE_C_COMPILER "${_HOST_SYSROOT}/usr/bin/${_TRIPLE}/${_TRIPLE}-gcc")
set(CMAKE_CXX_COMPILER "${_HOST_SYSROOT}/usr/bin/${_TRIPLE}/${_TRIPLE}-g++")

if(NOT EXISTS "${CMAKE_C_COMPILER}")
    message(FATAL_ERROR "Yocto C compiler not found: ${CMAKE_C_COMPILER}")
endif()

if(NOT EXISTS "${CMAKE_CXX_COMPILER}")
    message(FATAL_ERROR "Yocto C++ compiler not found: ${CMAKE_CXX_COMPILER}")
endif()

set(CMAKE_SYSROOT "${_TARGET_SYSROOT}")
set(CMAKE_C_FLAGS_INIT "${_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_ARCH_FLAGS}")

set(CMAKE_FIND_ROOT_PATH "${_TARGET_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(YOCTO_CROSS_TRIPLE "${_TRIPLE}" CACHE INTERNAL "" FORCE)
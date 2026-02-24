# FindAMLNN.cmake
# ---------------------------------------------------------------------------
# Locates the AMLNN nnsdk headers and libraries.
#
# Inputs (set before calling find_package):
#   AMLNN_HOME  – root of the amlnn-toolkit (contains nn_runtime/)
#                 May also be supplied as the environment variable AMLNN_HOME.
#
# Outputs:
#   AMLNN_INCLUDE_DIR  – path to nnsdk include dir
#   AMLNN_LIBRARY_DIR  – path to the ABI-specific library dir  (use link_directories)
#   AMLNN_LIBRARY      – "nnsdk"  (library name, no prefix/suffix)
#   AMLNN_FOUND
# ---------------------------------------------------------------------------

# Resolve AMLNN_HOME: CMake variable → env var → relative sibling fallbacks
if(NOT AMLNN_HOME)
    if(DEFINED ENV{AMLNN_HOME} AND NOT "$ENV{AMLNN_HOME}" STREQUAL "")
        set(AMLNN_HOME "$ENV{AMLNN_HOME}")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/../../../../../amlnn-toolkit/nn_runtime")
        set(AMLNN_HOME "${CMAKE_SOURCE_DIR}/../../../../../amlnn-toolkit")
    endif()
endif()

if(NOT AMLNN_HOME)
    message(FATAL_ERROR
        "AMLNN_HOME not found.\n"
        "Please set the AMLNN_HOME environment variable (or CMake variable) "
        "to the root of the amlnn-toolkit directory, e.g.:\n"
        "  export AMLNN_HOME=/path/to/amlnn-toolkit\n"
        "  cmake ... -DAMLNN_HOME=/path/to/amlnn-toolkit")
endif()

get_filename_component(AMLNN_HOME "${AMLNN_HOME}" ABSOLUTE)

set(AMLNN_NNSDK_ROOT "${AMLNN_HOME}/nn_runtime/nnsdk")
set(AMLNN_INCLUDE_DIR "${AMLNN_NNSDK_ROOT}/include")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(AMLNN_LIBRARY_DIR "${AMLNN_NNSDK_ROOT}/android/arm64-v8a")
    else()
        set(AMLNN_LIBRARY_DIR "${AMLNN_NNSDK_ROOT}/android/armeabi-v7a")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(AMLNN_LIBRARY_DIR "${AMLNN_NNSDK_ROOT}/linux/yocto/aarch64-poky-linux")
else()
    set(AMLNN_LIBRARY_DIR "${AMLNN_NNSDK_ROOT}/linux/yocto/aarch64-poky-linux")
endif()

set(AMLNN_LIBRARY "nnsdk")

# Validate paths
if(NOT EXISTS "${AMLNN_INCLUDE_DIR}")
    message(FATAL_ERROR "AMLNN include dir not found: ${AMLNN_INCLUDE_DIR}\n(AMLNN_HOME=${AMLNN_HOME})")
endif()
if(NOT EXISTS "${AMLNN_LIBRARY_DIR}")
    message(FATAL_ERROR "AMLNN library dir not found: ${AMLNN_LIBRARY_DIR}\n(AMLNN_HOME=${AMLNN_HOME})")
endif()

set(AMLNN_FOUND TRUE)
message(STATUS "Found AMLNN: ${AMLNN_HOME}")
message(STATUS "  AMLNN_INCLUDE_DIR: ${AMLNN_INCLUDE_DIR}")
message(STATUS "  AMLNN_LIBRARY_DIR: ${AMLNN_LIBRARY_DIR}")

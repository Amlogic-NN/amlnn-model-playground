# FindAMLLLM.cmake
# ---------------------------------------------------------------------------
# Locates the AMLNN LLMSDK headers and libraries.
#
# Inputs (set before calling find_package):
#   AMLNN_HOME  – root of the amlnn-toolkit/amlnn_runtime (contains llm_runtime/)
#                 May also be supplied as the environment variable AMLNN_HOME.
#
# Outputs:
#   AMLNN_LLM_INCLUDE_DIR  – path to llmsdk include dir
#   AMLNN_LLM_LIBRARY_DIR  – path to the ABI-specific library dir  (use link_directories)
#   AMLNN_LLM_LIBRARY      – "llmsdk"  (library name, no prefix/suffix)
#   AMLNN_LLM_FOUND
# ---------------------------------------------------------------------------

# Resolve AMLNN_HOME: CMake variable → env var → relative sibling fallbacks
if(NOT AMLNN_HOME)
    if(DEFINED ENV{AMLNN_HOME} AND NOT "$ENV{AMLNN_HOME}" STREQUAL "")
        set(AMLNN_HOME "$ENV{AMLNN_HOME}")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/../../../../../amlnn-toolkit/llm_runtime")
        set(AMLNN_HOME "${CMAKE_SOURCE_DIR}/../../../../../amlnn-toolkit")
    endif()
endif()

if(NOT AMLNN_HOME)
    message(FATAL_ERROR
        "AMLNN_HOME not found.\n"
        "Please set the AMLNN_HOME environment variable (or CMake variable) "
        "to the root of the amlnn-toolkit directory, e.g.:\n"
        "  export AMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime\n"
        "  cmake ... -DAMLNN_HOME=/path/to/amlnn-toolkit/amlnn_runtime")
endif()

get_filename_component(AMLNN_HOME "${AMLNN_HOME}" ABSOLUTE)

set(AMLNN_LLMSDK_ROOT "${AMLNN_HOME}/llm_runtime/llmsdk")
set(AMLNN_LLM_INCLUDE_DIR "${AMLNN_LLMSDK_ROOT}/include")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(AMLNN_LLM_LIBRARY_DIR "${AMLNN_LLMSDK_ROOT}/android/lib64")
    else()
        set(AMLNN_LLM_LIBRARY_DIR "${AMLNN_LLMSDK_ROOT}/android/lib32")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(DEFINED YOCTO_SDK_ROOT)
        if(DEFINED ARCH_BITS AND ARCH_BITS STREQUAL "32")
            set(AMLNN_LLM_LIBRARY_DIR "${AMLNN_LLMSDK_ROOT}/linux/yocto/lib32")
        else()
            set(AMLNN_LLM_LIBRARY_DIR "${AMLNN_LLMSDK_ROOT}/linux/yocto/lib64")
        endif()
    else()
        # TODO: libraries do not exist yet.
        if(DEFINED ARCH_BITS AND ARCH_BITS STREQUAL "32")
            set(AMLNN_LLM_LIBRARY_DIR "${AMLNN_LLMSDK_ROOT}/linux/buildroot/lib32")
        else()
            set(AMLNN_LLM_LIBRARY_DIR "${AMLNN_LLMSDK_ROOT}/linux/buildroot/lib64")
        endif()
    endif()

endif()

set(AMLNN_LLM_LIBRARY "llmsdk")

# Validate paths
if(NOT EXISTS "${AMLNN_LLM_INCLUDE_DIR}")
    message(FATAL_ERROR "AMLNN include dir not found: ${AMLNN_LLM_INCLUDE_DIR}\n(AMLNN_HOME=${AMLNN_HOME})")
endif()
if(NOT EXISTS "${AMLNN_LLM_LIBRARY_DIR}")
    message(FATAL_ERROR "AMLNN library dir not found: ${AMLNN_LLM_LIBRARY_DIR}\n(AMLNN_HOME=${AMLNN_HOME})")
endif()

set(AMLNN_LLM_FOUND TRUE)
message(STATUS "Found AMLNN: ${AMLNN_HOME}")
message(STATUS "  AMLNN_LLM_INCLUDE_DIR: ${AMLNN_LLM_INCLUDE_DIR}")
message(STATUS "  AMLNN_LLM_LIBRARY_DIR: ${AMLNN_LLM_LIBRARY_DIR}")

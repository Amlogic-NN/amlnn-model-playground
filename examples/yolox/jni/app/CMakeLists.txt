cmake_minimum_required(VERSION 3.22.1)

project(nnsdk_jni)

add_library(
    nnsdk_jni
    SHARED
    src/main/jni/nnsdk_jni.cpp
)

target_include_directories(
    nnsdk_jni
    PRIVATE
    src/main/jni
)

find_library(log-lib log)

target_link_libraries(
    nnsdk_jni
    PRIVATE
    ${log-lib}
    dl
)

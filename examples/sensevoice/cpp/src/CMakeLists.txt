cmake_minimum_required(VERSION 3.14)
project(sensevoice CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SENSEVOICE_BUILD_DEMO "Build sensevoice demo executable" ON)

set(SENSEVOICE_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../../../cmake")
find_package(AMLNN REQUIRED)
include_directories(${AMLNN_INCLUDE_DIR})
link_directories(${AMLNN_LIBRARY_DIR})

include(${SENSEVOICE_CPP_ROOT}/cmake/kaldi-native-fbank.cmake)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  link_libraries(log)
endif()

add_library(sensevoice STATIC
  sense_voice_engine.cpp
  feature_extractor.cpp
  adla_model.cpp
  ctc_decoder.cpp
  symbol_table.cpp
)

target_include_directories(sensevoice
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sensevoice
  PUBLIC
    kaldi-native-fbank-core
    ${AMLNN_LIBRARY}
)

if(SENSEVOICE_BUILD_DEMO)
  add_executable(sensevoice_demo
    main.cpp
    wav_reader.cpp
  )
  target_link_libraries(sensevoice_demo PRIVATE sensevoice dl m)
endif()

install(TARGETS sensevoice
  ARCHIVE DESTINATION lib
)
install(FILES sense_voice.h DESTINATION include)

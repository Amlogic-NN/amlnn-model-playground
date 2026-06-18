function(download_kaldi_native_fbank)
  include(FetchContent)

  set(kaldi_native_fbank_URL "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.22.3.tar.gz")
  set(kaldi_native_fbank_URL2 "https://hf-mirror.com/csukuangfj/sherpa-ncnn-cmake-deps/resolve/main/kaldi-native-fbank-1.22.3.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=9176cc66fc7ce1edf85cf355b06e320c57db6297df74277f575183468893cf61")

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-native-fbank-1.22.3.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-native-fbank-1.22.3.tar.gz
    ${CMAKE_SOURCE_DIR}/../kaldi-native-fbank-1.22.3.tar.gz
    ${CMAKE_SOURCE_DIR}/../../kaldi-native-fbank-1.22.3.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-native-fbank-1.22.3.tar.gz
    /tmp/kaldi-native-fbank-1.22.3.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_fbank_URL "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_fbank_URL}" kaldi_native_fbank_URL)
      set(kaldi_native_fbank_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_native_fbank
    URL
      ${kaldi_native_fbank_URL}
      ${kaldi_native_fbank_URL2}
    URL_HASH ${kaldi_native_fbank_HASH}
  )

  FetchContent_GetProperties(kaldi_native_fbank)
  if(NOT kaldi_native_fbank_POPULATED)
    FetchContent_Populate(kaldi_native_fbank)
  endif()

  set(_build_shared_libs_bak "${BUILD_SHARED_LIBS}")
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory(${kaldi_native_fbank_SOURCE_DIR} ${kaldi_native_fbank_BINARY_DIR} EXCLUDE_FROM_ALL)
  set(BUILD_SHARED_LIBS "${_build_shared_libs_bak}")

  target_include_directories(kaldi-native-fbank-core
    INTERFACE
      ${kaldi_native_fbank_SOURCE_DIR}
  )
endfunction()

download_kaldi_native_fbank()

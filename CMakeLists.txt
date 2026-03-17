cmake_minimum_required(VERSION 3.28)

project(playground_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.10.0%2Bcu130.zip" CACHE STRING "Download URL for the libtorch archive")
set(TORCH_ROOT "${CMAKE_SOURCE_DIR}/libtorch" CACHE PATH "Path to the libtorch distribution")
option(DOWNLOAD_LIBTORCH "Download libtorch into the project folder when it is missing" ON)
set(TORCH_CXX11_ABI "1" CACHE STRING "libtorch C++11 ABI setting")
option(ENABLE_CUDA_KERNEL_DEBUG "Build CUDA device code with kernel debug information" ON)

if(DOWNLOAD_LIBTORCH AND NOT EXISTS "${TORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
  set(LIBTORCH_ARCHIVE_DIR "${CMAKE_BINARY_DIR}/downloads")
  set(LIBTORCH_ARCHIVE_PATH "${LIBTORCH_ARCHIVE_DIR}/libtorch.zip")

  file(MAKE_DIRECTORY "${LIBTORCH_ARCHIVE_DIR}")

  message(STATUS "libtorch not found at ${TORCH_ROOT}")
  message(STATUS "Downloading libtorch from ${LIBTORCH_URL}")
  file(DOWNLOAD
    "${LIBTORCH_URL}"
    "${LIBTORCH_ARCHIVE_PATH}"
    SHOW_PROGRESS
    STATUS LIBTORCH_DOWNLOAD_STATUS
    TLS_VERIFY ON
  )
  list(GET LIBTORCH_DOWNLOAD_STATUS 0 LIBTORCH_DOWNLOAD_CODE)
  list(GET LIBTORCH_DOWNLOAD_STATUS 1 LIBTORCH_DOWNLOAD_MESSAGE)
  if(NOT LIBTORCH_DOWNLOAD_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to download libtorch: ${LIBTORCH_DOWNLOAD_MESSAGE}")
  endif()

  file(REMOVE_RECURSE "${TORCH_ROOT}")
  file(ARCHIVE_EXTRACT INPUT "${LIBTORCH_ARCHIVE_PATH}" DESTINATION "${CMAKE_SOURCE_DIR}")

  if(NOT EXISTS "${TORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
    message(FATAL_ERROR "libtorch was downloaded, but TorchConfig.cmake was not found in ${TORCH_ROOT}")
  endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

set(TORCH_INCLUDE_DIRS
  "${TORCH_ROOT}/include"
  "${TORCH_ROOT}/include/torch/csrc/api/include"
)

set(TORCH_LIBS
  "${TORCH_ROOT}/lib/libtorch.so"
  "${TORCH_ROOT}/lib/libtorch_cpu.so"
  "${TORCH_ROOT}/lib/libtorch_cuda.so"
  "${TORCH_ROOT}/lib/libc10.so"
  "${TORCH_ROOT}/lib/libc10_cuda.so"
)

foreach(TORCH_REQUIRED_FILE IN LISTS TORCH_INCLUDE_DIRS TORCH_LIBS)
  if(NOT EXISTS "${TORCH_REQUIRED_FILE}")
    message(FATAL_ERROR "Required libtorch file not found: ${TORCH_REQUIRED_FILE}")
  endif()
endforeach()

function(configure_playground_target target_name source_file)
  add_executable(${target_name} "${source_file}")

  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${TORCH_ROOT}/lib"
  )

  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0;-g3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter,-fPIC>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>,$<BOOL:${ENABLE_CUDA_KERNEL_DEBUG}>>:-O0;-g;-G>
  )

  target_compile_definitions(${target_name} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
  )

  target_include_directories(${target_name} PRIVATE
    ${TORCH_INCLUDE_DIRS}
  )

  target_link_libraries(${target_name} PRIVATE
    CUDA::cudart
    ${TORCH_LIBS}
  )

  target_link_options(${target_name} PRIVATE
    "LINKER:--no-as-needed"
  )
endfunction()

file(GLOB PLAYGROUND_SOURCE_FILES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cu"
  "${CMAKE_SOURCE_DIR}/examples/*.cu"
)

if(NOT PLAYGROUND_SOURCE_FILES)
  message(FATAL_ERROR "No runnable .cu files found in ${CMAKE_SOURCE_DIR}/src or ${CMAKE_SOURCE_DIR}/examples")
endif()

set(PLAYGROUND_DEFAULT_TARGET "")
foreach(PLAYGROUND_SOURCE IN LISTS PLAYGROUND_SOURCE_FILES)
  get_filename_component(PLAYGROUND_NAME "${PLAYGROUND_SOURCE}" NAME_WE)
  configure_playground_target("${PLAYGROUND_NAME}" "${PLAYGROUND_SOURCE}")
  if(PLAYGROUND_DEFAULT_TARGET STREQUAL "")
    set(PLAYGROUND_DEFAULT_TARGET "${PLAYGROUND_NAME}")
  endif()
endforeach()

add_custom_target(run
  COMMAND ${CMAKE_COMMAND} -E env
    "LD_LIBRARY_PATH=${TORCH_ROOT}/lib:$ENV{LD_LIBRARY_PATH}"
    $<TARGET_FILE:${PLAYGROUND_DEFAULT_TARGET}>
  DEPENDS ${PLAYGROUND_DEFAULT_TARGET}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  USES_TERMINAL
)

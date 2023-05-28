cmake_minimum_required(VERSION 3.20)

project(vs-ort VERSION 3.0 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
set(ONNX_RUNTIME_API_DIRECTORY "" CACHE PATH "Path to ONNX API headers")
set(ONNX_RUNTIME_LIB_DIRECTORY "" CACHE PATH "Path to ONNX Runtime libraries")

set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA backend")
set(ENABLE_DML OFF CACHE BOOL "Enable DirectML backend")

find_package(protobuf REQUIRED CONFIG)
find_package(ONNX REQUIRED CONFIG)

add_library(vsort SHARED
    vs_onnxruntime.cpp
    win32.cpp
    ../common/onnx_utils.cpp
    ../common/convert_float_to_float16.cpp
)

target_include_directories(vsort PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${ONNX_INCLUDE_DIRS}
    ${ONNX_RUNTIME_API_DIRECTORY}
)

target_link_directories(vsort PRIVATE
    ${ONNX_RUNTIME_LIB_DIRECTORY}
)

set_target_properties(vsort PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    set_target_properties(vsort PROPERTIES CXX_STANDARD 20)
endif()

target_link_libraries(vsort PRIVATE onnx onnxruntime)

if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_compile_definitions(ENABLE_CUDA)
    target_include_directories(vsort PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(vsort PRIVATE CUDA::cudart_static)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_link_options(vsort PRIVATE "/DELAYLOAD:onnxruntime.dll" "delayimp.lib")
    endif()
endif()

if (ENABLE_DML)
    add_compile_definitions(ENABLE_DML)
endif()

target_include_directories(vsort PUBLIC
    "${PROJECT_BINARY_DIR}"
)

find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE VCS_TAG
)
string(STRIP ${VCS_TAG} VCS_TAG)
configure_file(config.h.in config.h)

install(TARGETS vsort
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

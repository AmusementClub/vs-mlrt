cmake_minimum_required(VERSION 3.20)

project(vs-ort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
set(ONNX_RUNTIME_API_DIRECTORY "" CACHE PATH "Path to ONNX API headers")
set(ONNX_RUNTIME_LIB_DIRECTORY "" CACHE PATH "Path to ONNX Runtime libraries")

find_package(protobuf REQUIRED CONFIG)
find_package(ONNX REQUIRED CONFIG)

add_library(vsort SHARED vs_onnxruntime.cpp win32.cpp)

target_include_directories(vsort PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${ONNX_INCLUDE_DIRS}
    ${ONNX_RUNTIME_API_DIRECTORY})

target_link_directories(vsort PRIVATE
    ${ONNX_RUNTIME_LIB_DIRECTORY})

set_target_properties(vsort PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(vsort PRIVATE onnx onnxruntime)

if (WIN32)
    target_link_options(vsort PRIVATE "/DELAYLOAD:onnxruntime.dll" "delayimp.lib")
endif()

install(TARGETS vsort
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

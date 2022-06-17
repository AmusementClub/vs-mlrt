cmake_minimum_required(VERSION 3.20)

project(vs-ov VERSION 3.0 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
set(ENABLE_VISUALIZATION OFF CACHE BOOL "Enable support for network visualization")

find_package(OpenVINO REQUIRED CONFIG)
find_package(InferenceEngine REQUIRED CONFIG)

add_library(vsov SHARED
    vs_openvino.cpp
    win32.cpp
    ../common/onnx_utils.cpp
    ../common/convert_float_to_float16.cpp
)

target_compile_definitions(vsov PRIVATE HAVE_ONNX_FP16_RESIZE)

if (ENABLE_VISUALIZATION)
    target_compile_definitions(vsov PRIVATE ENABLE_VISUALIZATION)
endif()

target_include_directories(vsov PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${ONNX_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
)

target_link_libraries(vsov PRIVATE ${InferenceEngine_LIBRARIES} openvino::onnx)

set_target_properties(vsov PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if (WIN32)
    target_link_options(vsov PRIVATE "/DELAYLOAD:tbb.dll" "delayimp.lib")
endif()

target_include_directories(vsov PUBLIC
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

install(TARGETS vsov
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

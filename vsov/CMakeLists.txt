cmake_minimum_required(VERSION 3.20)

project(vs-ov VERSION 3.0 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
set(ENABLE_VISUALIZATION OFF CACHE BOOL "Enable support for network visualization")
set(WIN32_SHARED_OPENVINO OFF CACHE BOOL "Build for win32 with shared openvino library")

find_package(OpenVINO REQUIRED CONFIG)

add_library(vsov SHARED
    vs_openvino.cpp
    win32.cpp
    ../common/onnx_utils.cpp
    ../common/convert_float_to_float16.cpp
)

if(ENABLE_VISUALIZATION)
    target_compile_definitions(vsov PRIVATE ENABLE_VISUALIZATION)
endif()

if(WIN32_SHARED_OPENVINO)
    target_compile_definitions(vsov PRIVATE WIN32_SHARED_OPENVINO)
endif()

if(WIN32_SHARED_OPENVINO)
    find_package(protobuf REQUIRED CONFIG)
    find_package(ONNX REQUIRED CONFIG)
    target_link_libraries(vsov PRIVATE onnx)
else()
    target_link_libraries(vsov PRIVATE openvino::frontend::onnx)
endif()

target_include_directories(vsov PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${ONNX_INCLUDE_DIRS}
)

target_link_libraries(vsov PRIVATE openvino::runtime)

set_target_properties(vsov PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if (WIN32)
    if(WIN32_SHARED_OPENVINO)
        target_link_options(vsov PRIVATE "/DELAYLOAD:openvino.dll" "delayimp.lib")
    else()
        target_link_options(vsov PRIVATE "/DELAYLOAD:tbb.dll" "delayimp.lib")
    endif()
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

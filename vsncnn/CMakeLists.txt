cmake_minimum_required(VERSION 3.20)

project(vs-ncnn VERSION 3.0 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

find_package(protobuf REQUIRED CONFIG)
find_package(ONNX REQUIRED CONFIG)
find_package(ncnn REQUIRED CONFIG)

add_library(vsncnn SHARED vs_ncnn.cpp onnx2ncnn.cpp ../common/onnx_utils.cpp)

target_include_directories(vsncnn PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${ONNX_INCLUDE_DIRS}
)

target_link_libraries(vsncnn PRIVATE ncnn)

# https://github.com/onnx/onnx/commit/21bff4e55dcefecc069c679115baae6b00caa0d5
if (ONNX_VERSION VERSION_LESS 1.16.0)
    target_link_libraries(vsncnn PRIVATE onnx)
else()
    target_link_libraries(vsncnn PRIVATE ONNX::onnx)
endif()

set_target_properties(vsncnn PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if (CMAKE_CXX_STANDARD GREATER 17)
    set_target_properties(vsncnn PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD})
endif()

target_include_directories(vsncnn PUBLIC
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

install(TARGETS vsncnn
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

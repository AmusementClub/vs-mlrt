cmake_minimum_required(VERSION 3.20)

project(vs-trt VERSION 3.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
set(TENSORRT_HOME "" CACHE PATH "Path to TensorRT")
option(USE_NVINFER_PLUGIN "Initialize nvinfer_plugin" FALSE)
option(USE_NVINFER_PLUGIN_STATIC "Use static nvinfer_plugin" FALSE)
set(CUDNN_HOME "" CACHE PATH "Path to cuDNN")

FIND_PACKAGE(CUDAToolkit REQUIRED)

add_library(vstrt SHARED
    vs_tensorrt.cpp
    win32.cpp
)

target_include_directories(vstrt PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_HOME}/include
)

set_target_properties(vstrt PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_directories(vstrt PRIVATE ${TENSORRT_HOME}/lib)
target_link_libraries(vstrt PRIVATE CUDA::cudart_static nvinfer)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_link_options(vstrt PRIVATE "/DELAYLOAD:nvinfer.dll" "delayimp.lib")
endif()

if (USE_NVINFER_PLUGIN)
    add_definitions(-DUSE_NVINFER_PLUGIN)
    if (USE_NVINFER_PLUGIN_STATIC)
        target_link_libraries(vstrt PRIVATE nvinfer_plugin_static)

        find_library(CUDNN_LIB cudnn HINTS ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib)
        target_link_libraries(vstrt PRIVATE ${CUDNN_LIB} CUDA::cublas)

        if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_link_options(vstrt PRIVATE "/DELAYLOAD:cublas64_11.dll" "/DELAYLOAD:cudnn64_8.dll")
        endif()
    else()
        target_link_libraries(vstrt PRIVATE nvinfer_plugin)

        if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_link_options(vstrt PRIVATE "/DELAYLOAD:nvinfer_plugin.dll")
        endif()
    endif()
endif()

target_include_directories(vstrt PUBLIC
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

install(TARGETS vstrt
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

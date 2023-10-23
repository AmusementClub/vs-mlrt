cmake_minimum_required(VERSION 3.20)

project(trtexec LANGUAGES CXX)

set(TENSORRT_HOME "" CACHE PATH "Path to TensorRT")
set(CUDNN_HOME "" CACHE PATH "Path to cuDNN")

find_package(CUDAToolkit REQUIRED)

add_executable(trtexec
    trtexec.cpp
    logfile.cpp
    ../common/bfloat16.cpp
    ../common/logger.cpp
    ../common/sampleDevice.cpp
    ../common/sampleEngines.cpp
    ../common/sampleInference.cpp
    ../common/sampleOptions.cpp
    ../common/sampleReporting.cpp
    ../common/sampleUtils.cpp
    ../utils/fileLock.cpp
    ../utils/timingCache.cpp
)

target_include_directories(trtexec PRIVATE
    ../common
    ..
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_HOME}/include
)

target_link_directories(trtexec PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
    ${TENSORRT_HOME}/lib
)

find_library(CUDNN_LIB cudnn HINTS ${CUDNN_HOME} PATH_SUFFIXES lib lib/x64)

target_link_libraries(trtexec PRIVATE
    CUDA::cudart_static
    nvinfer
    nvinfer_plugin
    nvonnxparser
    CUDA::cublas
    ${CUDNN_LIB}
)

install(TARGETS trtexec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

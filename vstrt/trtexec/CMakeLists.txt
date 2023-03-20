cmake_minimum_required(VERSION 3.20)

project(trtexec LANGUAGES CXX)

set(TENSORRT_HOME "" CACHE PATH "Path to TensorRT")

find_package(CUDAToolkit REQUIRED)

add_executable(trtexec
    $<$<PLATFORM_ID:Windows>: trtexec.manifest>
    trtexec.cpp
    logfile.cpp
    ../common/logger.cpp
    ../common/sampleEngines.cpp
    ../common/sampleInference.cpp
    ../common/sampleOptions.cpp
    ../common/sampleReporting.cpp
    ../common/sampleUtils.cpp
)

target_include_directories(trtexec PRIVATE
    ../common
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_HOME}/include
)

target_link_directories(trtexec PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
    ${TENSORRT_HOME}/lib
)

target_link_libraries(trtexec PRIVATE
    CUDA::cudart_static
    nvinfer
    nvinfer_plugin
    nvparsers
    nvonnxparser
)

install(TARGETS trtexec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

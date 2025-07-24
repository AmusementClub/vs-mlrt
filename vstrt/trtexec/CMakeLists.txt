cmake_minimum_required(VERSION 3.20)

project(trtexec LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(trtexec
    $<$<PLATFORM_ID:Windows>: longpath.manifest>
    trtexec.cpp
    logfile.cpp
    ../common/bfloat16.cpp
    ../common/debugTensorWriter.cpp
    ../common/logger.cpp
    ../common/sampleDevice.cpp
    ../common/sampleEngines.cpp
    ../common/sampleInference.cpp
    ../common/sampleOptions.cpp
    ../common/sampleReporting.cpp
    ../common/sampleUtils.cpp
    ../utils/fileLock.cpp
    ../utils/cacheUtils.cpp
)

target_include_directories(trtexec PRIVATE
    ../common
    ..
    ../../include
)

target_link_libraries(trtexec PRIVATE CUDA::cudart_static)

install(TARGETS trtexec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

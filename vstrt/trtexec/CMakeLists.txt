cmake_minimum_required(VERSION 3.20)

project(trtexec LANGUAGES CXX CUDA)

add_executable(trtexec
    $<$<PLATFORM_ID:Windows>: longpath.manifest>
    trtexec.cpp
    trtexec_main.cpp
    logfile.cpp
    ../common/bigInt.cpp
    ../common/bfloat16.cpp
    ../common/debugTensorWriter.cpp
    ../common/globalTimerKernel.cu
    ../common/logger.cpp
    ../common/sampleDevice.cpp
    ../common/sampleEngines.cpp
    ../common/sampleInference.cpp
    ../common/sampleOptions.cpp
    ../common/sampleReporting.cpp
    ../common/sampleTuning.cpp
    ../common/sampleUtils.cpp
    ../../shared/utils/fileLock.cpp
    ../../shared/utils/cacheUtils.cpp
)

target_include_directories(trtexec PRIVATE
    ../common
    ..
    ../../include
    ../../shared
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

install(TARGETS trtexec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

cmake_minimum_required(VERSION 3.20)

project(trtexec LANGUAGES CXX CUDA)

add_executable(trtexec
    $<$<PLATFORM_ID:Windows>: longpath.manifest>
    trtexec.cpp
    trtexec_main.cpp
    logfile.cpp
    ../trtexecCommon/bfloat16.cpp
    ../trtexecCommon/bigInt.cpp
    ../trtexecCommon/debugTensorWriter.cpp
    ../trtexecCommon/globalTimerKernel.cu
    ../trtexecCommon/logger.cpp
    ../trtexecCommon/sampleDevice.cpp
    ../trtexecCommon/sampleEngines.cpp
    ../trtexecCommon/sampleInference.cpp
    ../trtexecCommon/sampleOptions.cpp
    ../trtexecCommon/sampleReporting.cpp
    ../trtexecCommon/sampleTuning.cpp
    ../trtexecCommon/sampleUtils.cpp
    ../../shared/utils/fileLock.cpp
    ../../shared/utils/cacheUtils.cpp
)

target_include_directories(trtexec PRIVATE
    ../trtexecCommon
    ..
    ../../include
    ../../shared
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_compile_definitions(trtexec PRIVATE TRT_BUILD_ONNX_PARSER=1)

install(TARGETS trtexec RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

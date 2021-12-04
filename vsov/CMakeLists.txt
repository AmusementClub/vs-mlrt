cmake_minimum_required(VERSION 3.20)

project(vs-ov VERSION 1.0 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

find_package(InferenceEngine REQUIRED)

add_library(vsov SHARED vs_openvino.cpp win32.cpp)

target_include_directories(vsov PRIVATE 
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${InferenceEngine_INCLUDE_DIRS}
)

target_link_libraries(vsov PRIVATE ${InferenceEngine_LIBRARIES})

set_target_properties(vsov PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if (WIN32)
    target_link_options(vsov PRIVATE "/DELAYLOAD:tbb.dll" "delayimp.lib")
endif()

install(TARGETS vsov
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

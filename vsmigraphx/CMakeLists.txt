cmake_minimum_required(VERSION 3.20)

project(vs-migraphx VERSION 3.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

find_package(migraphx REQUIRED CONFIG)
find_package(hip REQUIRED CONFIG)

add_library(vsmigraphx SHARED vs_migraphx.cpp)

target_include_directories(vsmigraphx PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
)

target_link_libraries(vsmigraphx PRIVATE migraphx::c hip::host)

set_target_properties(vsmigraphx PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE VCS_TAG
)
string(STRIP ${VCS_TAG} VCS_TAG)
configure_file(config.h.in config.h)
target_include_directories(vsmigraphx PUBLIC "${PROJECT_BINARY_DIR}")

install(TARGETS vsmigraphx
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

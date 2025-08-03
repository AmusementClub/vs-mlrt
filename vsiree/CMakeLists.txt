cmake_minimum_required(VERSION 3.21)

project(vs-iree LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
add_subdirectory(iree EXCLUDE_FROM_ALL)

add_library(vsiree SHARED vs_iree.cpp)

target_include_directories(vsiree PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})
set_target_properties(vsiree PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(vsiree PRIVATE iree_runtime_runtime)

find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE VCS_TAG
)
string(STRIP ${VCS_TAG} VCS_TAG)
configure_file(config.h.in config.h)
target_include_directories(vsiree PUBLIC "${PROJECT_BINARY_DIR}")

install(TARGETS vsiree
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

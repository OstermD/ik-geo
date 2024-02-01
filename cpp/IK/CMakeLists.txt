cmake_minimum_required(VERSION 3.9)
project(test_lib)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../subproblems build_sp)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(sources Spherical_IK.cpp General_IK.cpp)

add_library(SUBPROBLEMS_IK ${sources})
target_link_libraries(SUBPROBLEMS_IK SUBPROBLEMS)

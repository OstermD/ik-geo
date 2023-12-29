cmake_minimum_required(VERSION 3.9)
project(test_lib)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB sources *.cpp)

add_library(SUBPROBLEMS ${sources})

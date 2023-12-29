cmake_minimum_required(VERSION 3.9)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(test_lib)
add_subdirectory(subproblems)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 REQUIRED)

add_executable(${PROJECT_NAME} test_lib.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subproblems)
target_link_libraries(${PROJECT_NAME} SUBPROBLEMS)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
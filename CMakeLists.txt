cmake_minimum_required(VERSION 3.15)
project(FittingAlgorithms VERSION 0.0.1 LANGUAGES CXX)

option(FA_BUILD_DOCS "Build the documentation using Doxygen" ON)
option(FA_BUILD_TESTS "Build the tests using CTest" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_DOCS "Build the documentation using Doxygen" ON)
option(BUILD_TESTS "Build the tests using CTest" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP)

add_library(fittingalgorithms INTERFACE)

target_include_directories(fittingalgorithms INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src    
)
target_link_libraries(fittingalgorithms INTERFACE
  Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fittingalgorithms INTERFACE OpenMP::OpenMP_CXX)
endif()

if(FA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()

if(FA_BUILD_DOCS)
  add_subdirectory(docs)
endif()

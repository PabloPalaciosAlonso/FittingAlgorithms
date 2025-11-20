cmake_minimum_required(VERSION 3.15)

project(FittingAlgorithms VERSION 0.0.1 LANGUAGES CXX)

option(FA_BUILD_DOCS "Build the documentation using Doxygen" ON)
option(FA_BUILD_TESTS "Build the tests using CTest" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(fittingalgorithms INTERFACE)
target_include_directories(fittingalgorithms INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(FittingAlgorithms::fittingalgorithms ALIAS fittingalgorithms)

add_subdirectory(src)
target_link_libraries(fittingalgorithms INTERFACE GaussNewton ParallelTempering)

if(FA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif()

if(FA_BUILD_DOCS)
  add_subdirectory(docs)
endif()

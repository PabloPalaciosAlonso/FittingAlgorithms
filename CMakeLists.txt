cmake_minimum_required(VERSION 3.15)
project(FittingAlgorithms VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_DOCS "Build the documentation using Doxygen" ON)
option(BUILD_TESTS "Build the tests using CTest" ON)

find_package(Eigen3)
find_package(Boost COMPONENTS headers)
find_package(OpenMP)

set(USE_EIGEN OFF)
if(Eigen3_FOUND)
    message(STATUS "Eigen found: enabling USE_EIGEN")
    set(USE_EIGEN ON)
else()
    if(NOT Boost_FOUND)
        message(FATAL_ERROR "Neither Eigen nor Boost found")
    endif()
    message(STATUS "Eigen not found: using Boost uBLAS")
endif()

add_library(fittingalgorithms INTERFACE)

target_include_directories(fittingalgorithms INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Only defines USE_EIGEN if USE_EIGEN=ON
target_compile_definitions(fittingalgorithms INTERFACE
    $<$<BOOL:${USE_EIGEN}>:USE_EIGEN>
)

# Link backend
if(USE_EIGEN)
    target_link_libraries(fittingalgorithms INTERFACE Eigen3::Eigen)
else()
    target_link_libraries(fittingalgorithms INTERFACE Boost::headers)
endif()

# Optional OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(fittingalgorithms INTERFACE OpenMP::OpenMP_CXX)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

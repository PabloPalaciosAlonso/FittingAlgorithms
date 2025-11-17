cmake_minimum_required(VERSION 3.15)
project(FittingAlgorithms VERSION 0.1 LANGUAGES CXX)

#option(BUILD_PYTHON_WRAPPER "Build the Python wrapper using pybind11" OFF)
#option(BUILD_DOCS "Build the documentation using Doxygen" ON)
option(BUILD_TESTS "Build the tests using CTest" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)

add_subdirectory(src)

# find_package(Eigen3 REQUIRED)
# target_link_libraries(FittingAlgorithms PRIVATE Eigen3::Eigen)


# add_subdirectory(src)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif(
)
# if(BUILD_DOCS)
#   add_subdirectory(docs)
# endif()

# if(BUILD_PYTHON_WRAPPER)
#   add_subdirectory(wrappers/python/src)
# endif()

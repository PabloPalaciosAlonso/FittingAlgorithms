cmake_minimum_required(VERSION 3.15)

project(FittingAlgorithms VERSION 0.0.1 LANGUAGES CXX)


#option(FA_BUILD_PYTHON_WRAPPER "Build the Python wrapper using pybind11" OFF)
option(FA_BUILD_DOCS "Build the documentation using Doxygen" ON)
option(FA_BUILD_TESTS "Build the tests using CTest" ON)
set(FA_CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD ${FA_CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()



# Create interface target for consumers
add_library(fittingalgorithms INTERFACE)
target_include_directories(fittingalgorithms INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(FittingAlgorithms::fittingalgorithms ALIAS fittingalgorithms)

# Add actual library sources
add_subdirectory(src)

# Link implementation libraries to interface target
target_link_libraries(fittingalgorithms INTERFACE GaussNewton ParallelTempering)



if(FA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif()



if(FA_BUILD_DOCS)
  add_subdirectory(docs)
endif()


# Install rules for fittingalgorithms
install(TARGETS fittingalgorithms EXPORT fittingalgorithmsTargets)
install(EXPORT fittingalgorithmsTargets
  FILE fittingalgorithmsTargets.cmake
  NAMESPACE fittingalgorithms::
  DESTINATION lib/cmake/fittingalgorithms
)
install(DIRECTORY include/ DESTINATION include)

# Config file for find_package
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fittingalgorithmsConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/fittingalgorithmsConfig.cmake
  INSTALL_DESTINATION lib/cmake/fittingalgorithms
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fittingalgorithmsConfig.cmake
  DESTINATION lib/cmake/fittingalgorithms
)

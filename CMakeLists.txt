cmake_minimum_required(VERSION 3.15)
project(FittingAlgorithms VERSION 0.1.0 LANGUAGES CXX)

option(FA_BUILD_DOCS "Build the documentation using Doxygen" ON)
option(FA_BUILD_TESTS "Build the tests using CTest" ON)
option(FA_USE_EIGEN "Use Eigen library if found" ON)
option(FA_USE_OPENMP "Enable OpenMP parallelization" ON)
option(FA_BUILD_EXAMPLES "Build the usage examples" ON)

if (EMSCRIPTEN)
	# Avoid system-injected linker flags
	set(CMAKE_EXE_LINKER_FLAGS "")
	set(CMAKE_SHARED_LINKER_FLAGS "")
	set(CMAKE_MODULE_LINKER_FLAGS "")
	set(CMAKE_CXX_FLAGS "")
	set(CMAKE_C_FLAGS "")
  # Web builds run single-threaded; disable OpenMP to avoid unresolved symbols
  set(FA_USE_OPENMP OFF)
  set(FA_USE_EIGEN OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



if(NOT EMSCRIPTEN)
  find_package(Boost)
else()
  # Emscripten will bring its own Boost headers
  set(Boost_FOUND TRUE)
  add_library(Boost INTERFACE)
  target_compile_options(Boost INTERFACE "--use-port=boost_headers")
  add_library(Boost::headers ALIAS Boost)
endif()
if(FA_USE_OPENMP)
  find_package(OpenMP)
endif()
if(FA_USE_EIGEN)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
elseif(NOT Boost_FOUND)
  message(FATAL_ERROR "Boost library is required if Eigen is not used.")
endif()

add_library(fittingalgorithms INTERFACE)

target_include_directories(fittingalgorithms INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

if(FA_USE_EIGEN)
    target_link_libraries(fittingalgorithms INTERFACE Eigen3::Eigen)
  else()
    target_compile_options(fittingalgorithms INTERFACE -Wno-deprecated-declarations)
    target_link_libraries(fittingalgorithms INTERFACE Boost::headers)
endif()

if(FA_USE_OPENMP AND OpenMP_CXX_FOUND)
  target_compile_definitions(fittingalgorithms INTERFACE FA_OPENMP_ENABLED=1)
  target_link_libraries(fittingalgorithms INTERFACE OpenMP::OpenMP_CXX)
endif()

add_library(FittingAlgorithms::fittingalgorithms ALIAS fittingalgorithms)

if(FA_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()

if(NOT EMSCRIPTEN)
  if(FA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
  endif()

  if(FA_BUILD_DOCS)
    add_subdirectory(docs)
  endif()
endif()

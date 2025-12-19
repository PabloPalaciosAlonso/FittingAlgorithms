include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# EIGEN VARIANT
add_executable(test_GN_eigen test_GaussNewton.cpp)
target_compile_definitions(test_GN_eigen PRIVATE USE_EIGEN)
target_link_libraries(test_GN_eigen PRIVATE fittingalgorithms GTest::gtest_main)
gtest_discover_tests(test_GN_eigen)

# BOOST VARIANT
add_executable(test_GN_boost test_GaussNewton.cpp)
target_compile_definitions(test_GN_boost PRIVATE USE_EIGEN=0)
target_link_libraries(test_GN_boost PRIVATE fittingalgorithms GTest::gtest_main)
gtest_discover_tests(test_GN_boost)

add_executable(test_PT_eigen test_ParallelTempering.cpp)
target_compile_definitions(test_PT_eigen PRIVATE USE_EIGEN)
target_link_libraries(test_PT_eigen PRIVATE fittingalgorithms GTest::gtest_main)
gtest_discover_tests(test_PT_eigen)

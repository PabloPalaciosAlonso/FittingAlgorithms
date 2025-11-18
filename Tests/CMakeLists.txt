include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(TEST_SOURCES
  test_GaussNewton.cpp
  test_ParallelTempering.cpp
)

foreach(source_file IN LISTS TEST_SOURCES)
    add_executable(${source_file} ${source_file})
    target_link_libraries(${source_file} 
      PRIVATE 
      GaussNewton
      ParallelTempering
      GTest::gtest_main)
    gtest_discover_tests(${source_file}) 
endforeach()

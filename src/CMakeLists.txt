
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

add_library(GaussNewton STATIC GaussNewton.cpp)
target_include_directories(GaussNewton PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(GaussNewton PUBLIC Eigen3::Eigen)

add_library(ParallelTempering ParallelTempering.cpp)
target_include_directories(ParallelTempering PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelTempering PRIVATE OpenMP::OpenMP_CXX)
endif()

# Export implementation libraries for install
install(TARGETS GaussNewton ParallelTempering
    EXPORT fittingalgorithmsTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

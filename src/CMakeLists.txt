find_package(Eigen3 REQUIRED)
find_package(OpenMP)

add_library(GaussNewton GaussNewton.cpp)
target_link_libraries(GaussNewton
                      PUBLIC
                      Eigen3::Eigen)

add_library(ParallelTempering ParallelTempering.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelTempering PRIVATE OpenMP::OpenMP_CXX)
endif()

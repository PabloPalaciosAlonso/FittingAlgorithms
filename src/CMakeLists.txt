find_package(Eigen3 REQUIRED)

add_library(GaussNewton GaussNewton.cpp)
target_link_libraries(GaussNewton
                        PUBLIC
                        Eigen3::Eigen)

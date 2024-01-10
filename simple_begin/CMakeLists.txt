
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++17")

find_package(Ceres REQUIRED)

include_directories(${PROJECT_SOURCE_DIR} ${CSPARSE_INCLUDE_DIR})

add_executable(helloworld 01_helloworld.cc)
add_executable(powell 02_powell.cc)
add_executable(curve_fitting 03_curve_fitting.cc)
add_executable(robust_curve_fitting 04_robust_curve_fitting.cc)

target_link_libraries(helloworld ${CERES_LIBRARIES})
target_link_libraries(powell ${CERES_LIBRARIES})
target_link_libraries(curve_fitting ${CERES_LIBRARIES})
target_link_libraries(robust_curve_fitting ${CERES_LIBRARIES})


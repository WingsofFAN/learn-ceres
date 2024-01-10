
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++17")

find_package(Ceres REQUIRED)

include_directories(${PROJECT_SOURCE_DIR} ${CSPARSE_INCLUDE_DIR})

add_executable(bundle_adjuster bundle_adjuster.cc bal_problem.cc)
add_executable(simple_bal simple_bundle_adjuster.cc)
#add_executable(powell 02_powell.cc)
#add_executable(curve_fitting 03_curve_fitting.cc)
#add_executable(robust_curve_fitting 04_robust_curve_fitting.cc)

target_link_libraries(bundle_adjuster ${CERES_LIBRARIES})
target_link_libraries(simple_bal ${CERES_LIBRARIES})

#target_link_libraries(powell ${CERES_LIBRARIES})
#target_link_libraries(curve_fitting ${CERES_LIBRARIES})
#target_link_libraries(robust_curve_fitting ${CERES_LIBRARIES})


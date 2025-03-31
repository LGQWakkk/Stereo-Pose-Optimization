cmake_minimum_required(VERSION 3.10)
project(Project)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 3.3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp local.cpp)

target_link_libraries(main
    PRIVATE
        ${CERES_LIBRARIES}
        Eigen3::Eigen
        ${OpenCV_LIBS}
)
target_include_directories(main
    PRIVATE
        ${CERES_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

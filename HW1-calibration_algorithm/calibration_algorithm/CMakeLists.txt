cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(Eigen 3.0 REQUIRED)

add_executable(camera_calib
        main.cpp
 )
add_executable(test
        test.cpp
        )




cmake_minimum_required(VERSION 3.10)

project(stereo)

set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")


#set(OPENCV_DIR /home/q/workspace/opencv/install/lib/cmake/opencv)



find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen 3.0 REQUIRED)

find_package(Pangolin REQUIRED)

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        camera
        estimator
        visualizer
        feature
        sfm
)

add_executable(ba
        camera/camera.cpp
        # visualizer/visualizer.cpp
        visualizer/viewer.cpp
        estimator/essential_matrix.cpp
        estimator/triangulation.cpp
        estimator/utils.cpp
        estimator/ceres_optim.cpp
        estimator/EigenQuaternionParameterization.cpp
        feature/sift.cpp
        feature/match.cpp
        sfm/backend.cpp
        sfm/config.cpp
        sfm/dataset.cpp
        sfm/frame.cpp
        sfm/frontend.cpp
        sfm/map.cpp 
        sfm/mappoint.cpp 
        sfm/visual_sfm.cpp
        test.cpp
 )

target_link_libraries(ba PRIVATE
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${Pangolin_LIBRARIES}
)

# add_executable(iba
#         camera/camera.cpp
#         camera/sift.cpp
#         visualizer/visualizer.cpp
#         estimator/essential_matrix.cpp
#         estimator/triangulation.cpp
#         estimator/utils.cpp
#         main.cpp
#  )

# target_link_libraries(iba PRIVATE
#         ${Boost_LIBRARIES}
#         ${OpenCV_LIBS}
#         ${CERES_LIBRARIES}
#         ${Pangolin_LIBRARIES}
# )
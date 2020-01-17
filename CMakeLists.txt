cmake_minimum_required(VERSION 3.1)
# vcpkg
include(C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake)

project(fvp_calibration)
set (CMAKE_CXX_STANDARD 11)

add_subdirectory(./apriltag)

find_package(OpenCV REQUIRED)
find_package(Ceres CONFIG REQUIRED)

# opencv_demo
#add_executable(opencv_demo src/opencv_demo.cpp)
#target_link_libraries(opencv_demo apriltag ${OpenCV_LIBRARIES})

add_executable(calibration src/calibration.cpp src/ocam_functions.cpp src/optimizer.cpp src/angle_adjuster.cpp)
target_link_libraries(calibration apriltag ceres ${OpenCV_LIBRARIES})

#add_executable(optimization src/optimizer.cpp src/ocam_functions.cpp)
#target_link_libraries(optimization ceres ${OpenCV_LIBRARIES})

#add_executable(angleAdjuster src/angle_adjuster.cpp src/ocam_functions.cpp)
#target_link_libraries(angleAdjuster ${OpenCV_LIBRARIES})


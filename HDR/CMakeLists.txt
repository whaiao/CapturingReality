# CMakeList.txt : CMake project for HDR, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.23)
project(HDR)
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.7.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(opencv)

add_executable(HDR src/HDR.cpp src/HDR.h src/Tonemap.cpp src/Tonemap.h)

target_include_directories(HDR PRIVATE
        ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
        ${OPENCV_INCLUDE_INSTALL_PATH}
        ${OPENCV_MODULE_opencv_core_LOCATION}/include
        ${OPENCV_MODULE_opencv_videoio_LOCATION}/include
        ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
        ${OPENCV_MODULE_opencv_highgui_LOCATION}/include
        )

target_link_libraries(HDR opencv_core opencv_videoio opencv_imgcodecs opencv_highgui)

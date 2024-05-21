cmake_minimum_required(VERSION 3.20)
project(Rectangling)
add_executable(Rectangling Rectangling.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Rectangling ${OpenCV_LIBS})

add_subdirectory(glad)
target_link_libraries(Rectangling glad)


# SET(CMAKE_BUILD_TYPE "Release")
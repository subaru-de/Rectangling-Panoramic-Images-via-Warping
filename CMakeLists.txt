cmake_minimum_required(VERSION 3.20)
project(Rectangling)
add_executable(Rectangling Rectangling.cpp)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

target_link_libraries(Rectangling glfw OpenGL::GL)

find_package(OpenCV REQUIRED)
include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(Rectangling ${OpenCV_LIBS})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(
    ${Eigen_INCLUDE_DIRS}
)
# target_link_libraries(Rectangling ${Eigen_LIBRARIES})

add_subdirectory(glad)
target_link_libraries(Rectangling glad)


SET(CMAKE_BUILD_TYPE "Release")
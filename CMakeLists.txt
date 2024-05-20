cmake_minimum_required(VERSION 3.20)
project(Rectangling)
add_executable(Rectangling Rectangling.cpp)

add_subdirectory(glad)
target_link_libraries(Rectangling glad)
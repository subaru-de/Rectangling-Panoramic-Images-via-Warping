cmake_minimum_required(VERSION 3.20)
project(LSD)

add_library(LSD lsd.cpp lsd_cmd.cpp)

target_include_directories(
    LSD PUBLIC 
    "./"
)

# SET(CMAKE_BUILD_TYPE "Release")
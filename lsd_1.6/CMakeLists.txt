cmake_minimum_required(VERSION 3.20)
project(LSD)

add_library(LSD lsd.c)

target_include_directories(
    LSD PUBLIC 
    "${CMAKE_SOURCE_DIR}"
)

# SET(CMAKE_BUILD_TYPE "Release")
cmake_minimum_required(VERSION 4.0)

project(lib_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(lib_project
    src/main.cpp
)
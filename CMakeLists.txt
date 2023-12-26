cmake_minimum_required(VERSION 3.10)
project(CUDAProgramming LANGUAGES CXX CUDA)

# Set compiler options for the entire project
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(1_introduction)

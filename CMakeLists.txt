cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)
project(HighPerformanceComputing LANGUAGES CXX CUDA)

add_subdirectory(CUDA)
add_subdirectory(SharedMemory)
add_subdirectory(DistributedMemory)
add_subdirectory(Hybrid)


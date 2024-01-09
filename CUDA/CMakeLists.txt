cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)
project(CUDAProgramming LANGUAGES CXX CUDA)


# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
	

if(MSVC)
	set(CUDA_LIBRARIES cudart_static cublas cuda cusparse)
else()
	find_library(CUDA_RT cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	find_library(CUDA_BLAS cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	find_library(CUDA_CUDA cuda PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	find_library(CUDA_CUSPARSE cusparse PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
	set(CUDA_LIBRARIES ${CUDA_CUSPARSE} ${CUDA_RT} ${CUDA_BLAS} ${CUDA_CUDA} )
endif()

set(CMAKE_CUDA_FLAGS "-std c++17 ${CMAKE_CUDA_FLAGS}")

set(COMMON_FILES
    common/common.cpp	
)

add_library(commonlib ${COMMON_FILES})


add_subdirectory(1_Indexing)
add_subdirectory(2_Reduction)
add_subdirectory(3_Memory)
add_subdirectory(4_Evenets)
add_subdirectory(5_LinearAlgebra)





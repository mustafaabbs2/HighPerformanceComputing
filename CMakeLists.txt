cmake_minimum_required(VERSION 3.28)
project(CUDAProgramming LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)



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

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

add_subdirectory(1_introduction)

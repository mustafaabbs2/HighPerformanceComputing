cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(Reduction LANGUAGES CXX CUDA) #this will automatically compile CUDA sources with nvcc

message("Configuring Reduction..")

set(CUDA_SOURCES
    reduction.cu
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_SOURCES
    main.cpp
)

#Create static library (use shared for .dll)
add_library(${PROJECT_NAME} STATIC
    ${CUDA_SOURCES}
)

# Add executable target
add_executable(RunReduction
    ${EXECUTABLE_SOURCES}
)

# Set CUDA flags
    # Add include directories
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}  ${CUDA_LIBRARIES})


target_link_libraries(RunReduction PRIVATE ${PROJECT_NAME})






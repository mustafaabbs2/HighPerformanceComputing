cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(Introduction LANGUAGES CXX CUDA) #this will automatically compile CUDA sources with nvcc

message("Configuring Introduction..")

set(CUDA_SOURCES
    1_2_hello-world.cu
    1_3_add-2-numbers.cu
    1_4_get-device.cu
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

set(EXECS
    1_2_hello-world
    1_3_add-2-numbers
    1_4_get-device
)

foreach(EXEC ${EXECS})
    add_executable(${EXEC} ${EXEC}.cu)
    
    # Add include directories
    target_include_directories(${EXEC} PRIVATE 
    ${CUDA_INCLUDE_DIRS})

    target_link_libraries(${EXEC} 
    ${CUDA_LIBRARIES})

endforeach()

	



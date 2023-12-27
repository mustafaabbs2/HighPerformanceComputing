cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(Libraries LANGUAGES CXX CUDA) #this will automatically compile CUDA sources with nvcc

message("Configuring Libraries..")


# List of CUDA source files
set(CUDA_SOURCES
    5_1_cusparse.cu
    5_2_thrust.cu
    5_3_cublasmatrixmul.cu
    5_3_cublasvectoradd.cu
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

set(EXECS
    5_1_cusparse
    5_2_thrust
    5_3_cublasmatrixmul
    5_3_cublasvectoradd
)


foreach(EXEC ${EXECS})
    add_executable(${EXEC} ${EXEC}.cu)
    
    # Add include directories
    target_include_directories(${EXEC} PRIVATE 
    ${CUDA_INCLUDE_DIRS})

    target_link_libraries(${EXEC} 
    ${CUDA_LIBRARIES}
    commonlib
    )

endforeach()

	



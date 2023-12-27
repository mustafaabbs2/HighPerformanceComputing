cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(Events LANGUAGES CXX CUDA) #this will automatically compile CUDA sources with nvcc

message("Configuring Events..")


# List of CUDA source files
set(CUDA_SOURCES
    4_1_hello-world_timing.cu
    4_3_stream.cu
    4_4_simplenullstream.cu
    4_5_streams_sumarray.cu
    4_6_blocking_nonblocking.cu
    4_7_interstreamdependencies.cu
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

set(EXECS
    4_1_hello-world_timing
    4_3_stream
    4_4_simplenullstream
    4_5_streams_sumarray
    4_6_blocking_nonblocking
    4_7_interstreamdependencies
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

	



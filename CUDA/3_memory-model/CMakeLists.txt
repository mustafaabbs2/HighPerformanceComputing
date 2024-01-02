cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(MemoryModel LANGUAGES CXX CUDA) #this will automatically compile CUDA sources with nvcc

message("Configuring MemoryModel..")


# List of CUDA source files
set(CUDA_SOURCES
    3_1_intro-mem-model.cu
    3_2_registers.cu
    3_3_pinnedmem.cu
    3_4_zerocopymem.cu
    3_5_sumArray_unifiedmemory.cu
    3_6_globalVariable.cu
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

set(EXECS
    3_1_intro-mem-model
    3_2_registers
    3_3_pinnedmem
    3_4_zerocopymem
    3_5_sumArray_unifiedmemory
    3_6_globalVariable
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

	



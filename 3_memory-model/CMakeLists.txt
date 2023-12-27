cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(ExecutionModel LANGUAGES CXX CUDA) #this will automatically compile CUDA sources with nvcc

message("Configuring ExecutionModel..")

#2_10_profiling.cu -> needs correction

set(CUDA_SOURCES
    2_1_blockdim.cu
    2_2_thread-id.cu
    2_3_block-id.cu
    2_4_sum_array.cu
    2_6_sumArray-cpu-gpu.cu
    2_6_warp-id.cu
    2_7_uniqueId_1d.cu
    2_8_uniqueId_2d.cu
    2_9_uniqueId_3d.cu
    2_11_warp-divergence.cu
    2_12_warp-unroll.cu
    2_13_matrixmul_naive.cu
    2_14_matrixmul_tiled.cu
    2_15_sumreduction.cu
    2_16_sumreduction-working.cu
    2_17_sumreduction-nodivergence.cu
    2_18_sumreduction-nobankconflicts.cu
    2_19_sumreduction-reducethreads.cu
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

set(EXECS
    2_1_blockdim
    2_2_thread-id
    2_3_block-id
    2_4_sum_array
    2_6_sumArray-cpu-gpu
    2_6_warp-id
    2_7_uniqueId_1d
    2_8_uniqueId_2d
    2_9_uniqueId_3d
    2_11_warp-divergence
    2_12_warp-unroll
    2_13_matrixmul_naive
    2_14_matrixmul_tiled
    2_15_sumreduction
    2_16_sumreduction-working
    2_17_sumreduction-nodivergence
    2_18_sumreduction-nobankconflicts
    2_19_sumreduction-reducethreads
)

foreach(EXEC ${EXECS})
    add_executable(${EXEC} ${EXEC}.cu)
    
    # Add include directories
    target_include_directories(${EXEC} PRIVATE 
    ${CUDA_INCLUDE_DIRS})

    target_link_libraries(${EXEC} 
    ${CUDA_LIBRARIES})

endforeach()

	



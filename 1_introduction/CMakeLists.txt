project(1_introduction LANGUAGES CXX CUDA)

set(CUDA_SOURCES
    1_2_hello-world.cu
    1_3_add-2-numbers.cu
    1_4_get-device.cu
)

foreach(EXEC ${EXECS})
    add_executable(${EXEC})
endforeach()

# Add include directories
target_include_directories(1_introduction PRIVATE 
${CUDA_INCLUDE_DIRS})

target_link_libraries(1_introduction 
${CUDA_LIBRARIES}
)
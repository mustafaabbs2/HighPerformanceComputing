
set(CUDA_SOURCES
    1_2_hello-world.cu
    1_3_add-2-numbers.cu
    1_4_get-device.cu
)

foreach(EXEC ${EXECS})
    add_executable(${EXEC})
endforeach()

cmake_minimum_required(VERSION 3.5)

project(Hybrid LANGUAGES CXX) 

message("Configuring Hybrid..")

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CUDA_ON FALSE)
set(KOKKOS_ON FALSE)

if(KOKKOS_ON)

set(CPP_SOURCES
    kokkos.cpp
    DAGParallel.cpp
)

set(EXECUTABLE_SOURCES
    main.cpp
)


# Add executable target
add_executable(BenchmarkHybrid
    ${EXECUTABLE_SOURCES}
)

#Create static library (use shared for .dll)
add_library(${PROJECT_NAME} STATIC
    ${CPP_SOURCES}
)

# Define the variable kokkos_install_prefix with the desired value
if(WIN32)
    if(CUDA_ON)
        set(kokkos_install_prefix "D:\\Mustafa\\kokkos\\build-cuda-openmp")
    else()
        set(kokkos_install_prefix "D:\\Mustafa\\kokkos\\build-openmp")
    endif()
elseif(UNIX)
    if(CUDA_ON)
        set(kokkos_install_prefix "/home/mustafaabbs2/kokkos/build-cuda-openmp") #this has cuda for the device and pthreads for the host
        set(CMAKE_CXX_COMPILER "/home/mustafaabbs2/kokkos/kokkos/bin/nvcc_wrapper") 
        #All current versions of the NVCC compiler have shortcomings when used as the main compiler for a project, 
        #in particular when part of a complex build system.
        #For example, it does not understand most GCC command-line options,
        #which must be prepended by the -Xcompiler flag when calling NVCC
        set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -expt-relaxed-constexpr")
    else()
        #set(kokkos_install_prefix "/home/mustafaabbs2/kokkos/build-threads")
        set(kokkos_install_prefix "/home/mustafaabbs2/kokkos/build-openmp")
        set(CMAKE_CXX_COMPILER g++) 
    endif()
endif()


set(Kokkos_DIR ${kokkos_install_prefix})
# Add the -DKokkos_ROOT option to the compile definitions
set(Kokkos_DIR ${kokkos_install_prefix})

if(WIN32 OR UNIX)
    find_package(Kokkos REQUIRED)
    target_link_libraries(${PROJECT_NAME} Kokkos::kokkos)
else() #for apple
        message(FATAL_ERROR "MacOS currently unsupported")
endif()

target_link_libraries(BenchmarkHybrid PRIVATE ${PROJECT_NAME})

endif()

# Now, the stuff for the sycl executable on both windows and unix

set(SYCL_SOURCES
    sycl.cpp
)

if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker 
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()


# Add executable target
add_executable(BenchmarkSycl
    ${SYCL_SOURCES}
)



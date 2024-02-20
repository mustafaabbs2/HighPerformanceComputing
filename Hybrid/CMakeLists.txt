cmake_minimum_required(VERSION 3.5)

project(Hybrid LANGUAGES CXX) 

message("Configuring Hybrid..")

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)


if(UNIX)

    set(CPP_SOURCES
        kokkos.cpp
        DAGParallel.cpp
    )

    set(EXECUTABLE_SOURCES
        main.cpp
    )


    #Create static library (use shared for .dll)
    add_library(${PROJECT_NAME} STATIC
        ${CPP_SOURCES}
    )


    # Add executable target
    add_executable(BenchmarkHybrid
        ${EXECUTABLE_SOURCES}
    )

    # Define the variable kokkos_install_prefix with the desired value
    if(WIN32)
        set(kokkos_install_prefix "D:\\Mustafa\\kokkos\\build")
    elseif(UNIX)
        set(kokkos_install_prefix "/home/mustafaabbs2/kokkos/build")
        set(CMAKE_CXX_COMPILER g++)
    endif()

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



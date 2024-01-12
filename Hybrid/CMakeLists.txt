cmake_minimum_required(VERSION 3.5)

project(Hybrid LANGUAGES CXX) 

message("Configuring Hybrid..")

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

set(CPP_SOURCES
    kokkosCore.cpp
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
set(kokkos_install_prefix "D:\\Mustafa\\kokkos\\build")

# Add the -DKokkos_ROOT option to the compile definitions
set(Kokkos_DIR ${kokkos_install_prefix})

if(WIN32 OR UNIX)
    find_package(Kokkos REQUIRED)
    target_link_libraries(${PROJECT_NAME} Kokkos::kokkos)
else() #for apple
        message(FATAL_ERROR "MacOS currently unsupported")
endif()

target_link_libraries(BenchmarkHybrid PRIVATE ${PROJECT_NAME})





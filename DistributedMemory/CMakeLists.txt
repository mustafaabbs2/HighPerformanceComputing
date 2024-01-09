cmake_minimum_required(VERSION 3.5)

project(DistributedMemory LANGUAGES CXX) 

message("Configuring DistributedMemory..")

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

set(CPP_SOURCES
    Multiport.cpp
)

set(EXECUTABLE_SOURCES
    main.cpp
)


#Create static library (use shared for .dll)
add_library(${PROJECT_NAME} STATIC
    ${CPP_SOURCES}
)

# Add executable target
add_executable(BenchmarkMPI
    ${EXECUTABLE_SOURCES}
)


if(WIN32 OR UNIX)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME}  PUBLIC ${MPI_CXX_LIBRARIES})
        target_include_directories(${PROJECT_NAME}  PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "MPI not found. Please make sure to install it.")
    endif()
else() #for apple
        message(FATAL_ERROR "MacOS currently unsupported")
endif()

target_link_libraries(BenchmarkMPI PRIVATE ${PROJECT_NAME})





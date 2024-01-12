cmake_minimum_required(VERSION 3.5)

project(SharedMemory LANGUAGES CXX) 

message("Configuring SharedMemory..")

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

set(CPP_SOURCES
    GraphOptimization.cpp
    PiApproximation.cpp
)

set(EXECUTABLE_SOURCES
    main.cpp
)


#Create static library (use shared for .dll)
add_library(${PROJECT_NAME} STATIC
    ${CPP_SOURCES}
)

# Add executable target
add_executable(Benchmark
    ${EXECUTABLE_SOURCES}
)


if(WIN32 OR UNIX)
    # Find OpenMP package
    find_package(OpenMP REQUIRED)

    # Enable OpenMP
    if(OpenMP_CXX_FOUND)
        # Set OpenMP flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        if(UNIX)
            target_link_libraries(${PROJECT_NAME}  PUBLIC OpenMP::OpenMP_CXX tbb)
        else()
            target_link_libraries(${PROJECT_NAME}  PUBLIC OpenMP::OpenMP_CXX)
        endif()
    else()
        message(FATAL_ERROR "OpenMP not found. Please make sure to install it.")
    endif()
else() #for apple

    # Specify include directories
    include_directories(/opt/homebrew/opt/libomp/include)

    # Specify library directories
    link_directories(/opt/homebrew/opt/libomp/lib)

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib")

    # Specify the OpenMP library 
    target_link_libraries(${PROJECT_NAME} PUBLIC omp)

    # Enable OpenMP for C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")

endif()


target_link_libraries(Benchmark PRIVATE ${PROJECT_NAME})





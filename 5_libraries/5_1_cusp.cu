#include <thrust/version.h>
#include "cusp/version.h"
#include <iostream>
//need these CUDA headers
#include "cuda_runtime.h"
#include "cuda.h" //Need this for CUDA_VERSION and THRUST_MAJOR_VERSION etc 
#include "device_launch_parameters.h"

int main(void)
{
    int cuda_major =  CUDA_VERSION / 1000;
    int cuda_minor = (CUDA_VERSION % 1000) / 10;
    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;
    int cusp_major = CUSP_MAJOR_VERSION;
    int cusp_minor = CUSP_MINOR_VERSION;
    std::cout << "CUDA   v" << cuda_major   << "." << cuda_minor   << std::endl;
    std::cout << "Thrust v" << thrust_major << "." << thrust_minor << std::endl;
    std::cout << "Cusp   v" << cusp_major   << "." << cusp_minor   << std::endl;
    return 0;
}
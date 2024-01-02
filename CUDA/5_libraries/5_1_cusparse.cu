#include <iostream>
#include <thrust/version.h>
#include <cuda_runtime.h>
#include <cusparse.h>

int main(void)
{
    int cuda_major = CUDART_VERSION / 1000;
    int cuda_minor = (CUDART_VERSION % 1000) / 10;
    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Get cusparse version information
    int cusparse_version;
    cusparseGetVersion(handle, &cusparse_version);
    int cusparse_major = cusparse_version / 1000;
    int cusparse_minor = (cusparse_version % 1000) / 10;

    cusparseDestroy(handle);

    std::cout << "CUDA       v" << cuda_major << "." << cuda_minor << std::endl;
    std::cout << "Thrust     v" << thrust_major << "." << thrust_minor << std::endl;
    std::cout << "cusparse   v" << cusparse_major << "." << cusparse_minor << std::endl;

    return 0;
}

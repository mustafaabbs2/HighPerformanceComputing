#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <vector>

void printVersions()
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
}

void cublasVecAdd()
{
	int n = 1 << 4; //size of the vector
	float *h_a, *h_b, *h_c; //vectors a, b, c
	float *d_a, *d_b; //device vectors a, b

	//Allocate memory
	h_a = (float*)malloc(n * sizeof(float));
	h_b = (float*)malloc(n * sizeof(float));
	h_c = (float*)malloc(n * sizeof(float));

	cudaMalloc((void**)&d_a, n * sizeof(float));
	cudaMalloc((void**)&d_b, n * sizeof(float));

	// Seed for random number generation
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	for(size_t i = 0; i < n; i++)
	{
		h_a[i] = std::rand();
		h_b[i] = std::rand();
	}

	//create and initialize the cublas handle
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	//copy the host vectors to the device

	cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1); //the last element is the step size
	cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1); //the last element is the step size

	const float scale = 1.0f;
	cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1); //ax+b, set a to 1.0f

	cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

	cublasDestroy(handle);

	cudaFree(d_a);
	cudaFree(d_b);
	free(h_a);
	free(h_b);
}

void thrustReduce()
{
	thrust::device_vector<int> data(4);
	data[0] = 10;
	data[1] = 20;
	data[2] = 30;
	data[3] = 40;
	int sum = thrust::reduce(data.begin(), data.end());
	std::cout << "sum is " << sum << std::endl;
}

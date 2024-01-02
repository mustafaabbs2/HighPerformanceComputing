//C headers
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include <cublas_v2.h>
#include <math.h>

#include "../common/common.h"

int main()
{
	int n = 1 << 2; //size of the vector
	float *h_a, *h_b, *h_c; //vectors a, b, c
	float *d_a, *d_b; //device vectors a, b

	//Allocate memory
	h_a = (float*)malloc(n * sizeof(float));
	h_b = (float*)malloc(n * sizeof(float));
	h_c = (float*)malloc(n * sizeof(float));

	cudaMalloc((void**)&d_a, n * sizeof(float));
	cudaMalloc((void**)&d_b, n * sizeof(float));

	vector_init(h_a, n);
	vector_init(h_b, n);

	//create and initialize the cublas handle
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	//copy the host vectors to the device

	cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1); //the last element is the step size
	cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1); //the last element is the step size

	const float scale = 1.0f;
	cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1); //ax+b, set a to 1.0f

	cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

	verify_result(h_a, h_b, h_c, n);

	cublasDestroy(handle);

	cudaFree(d_a);
	cudaFree(d_b);
	free(h_a);
	free(h_b);

	return 0;
	// nvcc -o 5_3_cublasvectoradd -lcublas 5_3_cublasvectoradd.cu
}

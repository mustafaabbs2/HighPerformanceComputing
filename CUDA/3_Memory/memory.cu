#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

//helper functions
#define SHMEM_SIZE 256

#define CHECK(call)                                                                                \
	{                                                                                              \
		const cudaError_t error = call;                                                            \
		if(error != cudaSuccess)                                                                   \
		{                                                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                                 \
			fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));           \
			exit(1);                                                                               \
		}                                                                                          \
	}

//device functions
__global__ void sumArray(float* a, float* b, float* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
}

//Wrapper functions

void sumArray_(size_t blockSize)
{
	int size = 1 << 22;

	printf("Entered block size : %zd \n", blockSize);
	printf("Input size : %d \n", size);
	unsigned int byte_size = size * sizeof(float);

	float *A, *B, *ref, *C;

	//V0//////////////////////////
	C = (float*)malloc(byte_size);

	cudaMallocManaged((void**)&A, byte_size);
	cudaMallocManaged((void**)&B, byte_size);
	cudaMallocManaged((void**)&ref, byte_size);

	// Seed for random number generation
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	for(size_t i = 0; i < size; i++)
	{
		A[i] = std::rand();
		B[i] = std::rand();
	}

	dim3 block(blockSize);
	dim3 grid((size + block.x - 1) / block.x);

	std::cout << "Array Sum with Unified Memory" << std::endl;
	auto start_time = std::chrono::high_resolution_clock::now();

	sumArray<<<grid, block>>>(A, B, ref, size);

	auto end_time = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V0: " << duration.count() << " microseconds" << std::endl;

	free(C);
}



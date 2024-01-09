
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <cmath>

#define SIZE 256
#define SHMEM_SIZE                                                                                 \
	256 //size of shared memory in bytes - 256 threads in a block (shared memory is shared between a block)

// 256*4=1024

void initialize_vector(int* v, int n)
{
	for(int i = 0; i < n; i++)
	{
		v[i] = 1; //should be 65536
	}
}

void sum_vector(int* v, int n)
{
	int sum = 0;
	for(int i = 0; i < n; i++)
	{
		sum += v[i]; //should be 65536
	}
	printf("sum = %d\n", sum);
}

__device__ void sum_vector_device(int* v, int n)
{
	int sum = 0;
	for(int i = 0; i < n; i++)
	{
		sum += v[i]; //should be 65536
	}
	printf("sum from device = %d\n", sum);
}

void sharedMemory()
{

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Shared memory in bytes: %ld\n", prop.sharedMemPerBlock);
}

__global__ void sumReduction(int* v, int* v_r)
{

	__shared__ int partial_sum[SHMEM_SIZE]; //allocate shared memory

	int n = 256;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	partial_sum[threadIdx.x] = v[tid]; //load data of 256 threads into each partial sum array
	__syncthreads(); //wait for all threads to finish loading data

	for(int s = 1; s < blockDim.x; s *= 2)
	{
		int index = threadIdx.x;
		if(index % (2 * s) == 0)
		{
			partial_sum[index] += partial_sum[index + s]; //writing in shared memory
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
	{ //only if the thread is the first thread in the block (all threads access same partial sum)
		v_r[blockIdx.x] =
			partial_sum[0]; //write back to global memory the partial sum for each block
	}

	// if (tid == 0)
	// sum_vector_device(v_r, n);
}

int main()
{

	int n = 1 << 16; //65536;
	size_t bytes = n * sizeof(int);

	int *h_v, *h_v_r;
	int *d_v, *d_v_r, *d_v_r_2;

	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);
	cudaMalloc(&d_v_r_2, bytes);

	initialize_vector(h_v, n);

	int TB_SIZE = SIZE;

	int GRID_SIZE = (int)ceil(n / TB_SIZE);

	printf("Grid size: %d\n", GRID_SIZE);
	printf("TB_SIZE: %d\n", TB_SIZE);
	sharedMemory(); //print shared memory size

	// sum_vector(h_v, n);

	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice); //copy from host to device memory

	// break this into two kernels
	// launch 256 thread blocks, each with 256 threads
	// 256*256= 65536
	// There are 256 partial sums, for which another kernel is launched to compute the final sum

	sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
	// sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r_2);

	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyHostToDevice); //copy from device to host memory

	sum_vector(h_v_r, 256);

	printf("Accumulated sum: %d\n", h_v_r[0]);

	return 0;

	// nvcc -o 2_15_sumreduction 2_15_sumreduction.cu
}
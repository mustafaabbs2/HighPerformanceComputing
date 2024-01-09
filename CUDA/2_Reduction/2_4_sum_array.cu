#include <stdio.h>
#include <stdlib.h>
//need these CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Contains self written helper functions
#include "../common/common.h"

__global__ void sum_array(int* a, int* b, int* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
}

int main(void)
{

	int size = 10;
	int byte_size = size * sizeof(int);

	int *h_a, *h_b, *h_c;
	h_a = (int*)malloc(byte_size);
	h_b = (int*)malloc(byte_size);
	h_c = (int*)malloc(byte_size);

	for(int i = 0; i < size; i++)
	{
		h_a[i] = i + 1;
		h_b[i] = i + 1;
		h_c[i] = 0;
	}

	//allocating memory on the device
	int *d_a, *d_b, *d_c;
	CHECK(cudaMalloc((void**)&d_a, byte_size));
	CHECK(cudaMalloc((void**)&d_b, byte_size));
	CHECK(cudaMalloc((void**)&d_c, byte_size));

	CHECK(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_c, h_c, byte_size, cudaMemcpyHostToDevice));

	dim3 block(64); //block with 64 threads in each direction
	dim3 grid(2); //two blocks

	sum_array<<<grid, block>>>(d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	CHECK(cudaMemcpy(h_c, d_c, byte_size, cudaMemcpyDeviceToHost));

	for(int i = 0; i < size; i++)
	{
		printf("%d ", h_c[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	cudaDeviceReset();

	return 0;
}
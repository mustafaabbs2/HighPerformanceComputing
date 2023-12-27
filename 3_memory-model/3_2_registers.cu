//C headers
#include <stdio.h>
#include <stdlib.h>
//need these CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Contains error check
#include "../common/cuda_common.cuh"
//Contains self written helper functions
#include "../common/common.h"

__global__ void registers(int* results, int size)
{
	int gid;
	int x1 = 200;
	int x2 = 201;
	int x3 = 202;
	int x4 = x1 + x2 + x3;

	gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid < size)
	{
		results[gid] = x4;
	}
}

int main()
{
	int size = 1 << 2;

	int byte_size = sizeof(int) * size;

	int* h_temp = (int*)malloc(byte_size);
	int* d_results;

	gpuErrchk(cudaMalloc((void**)&d_results, byte_size));
	gpuErrchk(cudaMemset(d_results, 0, byte_size));

	dim3 block(128);
	dim3 grid((size + block.x - 1) / block.x);

	printf("Launching kernel, 1,2,3.... \n");

	registers<<<grid, block>>>(d_results, size);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(h_temp, d_results, byte_size, cudaMemcpyDeviceToHost));

	int sum = 0;

	for(int i = 0; i < size; i++)
	{
		sum = h_temp[i];
	}

	printf("The sum is: %d", sum);

	return 0;

	// nvcc --ptxas-options=-v 3_2_registers.cu
}
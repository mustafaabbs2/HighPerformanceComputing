#include <stdio.h>
#include <stdlib.h>
//need these CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Contains error check 
#include "../common/cuda_common.cuh"
//Contains self written helper functions
#include "../common/common.h"

__global__ void sumArrays(float *a, float *b, float *c, int size)
{

    int gid = blockIdx.x * blockDim.x + threadIdx.x; 

    if(gid < size)
	{
        c[gid] = a[gid] + b[gid];
    }
}

int main(int argc, char** argv)
{
	printf("Runing 1D grid \n");
	int size = 1 << 22;
	int block_size = 128;

	if (argc > 1)
		block_size = 1 << atoi(argv[1]);

	printf("Block size selected: %d \n", block_size);

	unsigned int byte_size = size * sizeof(float);

	printf("Input size : %d \n", size);

	float * h_a, *h_b, *h_ref;
	h_a = (float*)malloc(byte_size);
	h_b = (float*)malloc(byte_size);
	h_ref = (float*)malloc(byte_size);


	if (!h_a)
		printf("Host memory unallocated \n");

	for (size_t i = 0; i < size; i++)
	{
		h_a[i] = i % 10;
		h_b[i] = i % 7;
	}

	dim3 block(block_size);
	dim3 grid((size + block.x - 1) / block.x);

	printf("Kernel is lauched with grid(%d,%d,%d) and block(%d,%d,%d) \n",
		grid.x, grid.y, grid.z, block.x, block.y, block.z);

	float *d_a, *d_b, *d_c;

	gpuErrchk(cudaMalloc((void**)&d_a, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_b, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_c, byte_size));
	gpuErrchk(cudaMemset(d_c, 0, byte_size));

	gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));

	sumArrays << <grid, block >> > (d_a, d_b, d_c, size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_ref, d_c, byte_size, cudaMemcpyDeviceToHost));

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	free(h_ref);
	free(h_b);
	free(h_a);

    // nvprof --metrics gld_efficiency,gld_throughput,gld_transactions,gld_transactions_per_request .\a.exe <x>
}
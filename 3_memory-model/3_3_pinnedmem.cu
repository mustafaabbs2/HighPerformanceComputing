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

int main()
{
	int size = 1 << 25;

	int byte_size = sizeof(int) * size;

	//pageable -> pinned
	// int *h_temp = (int*) malloc(byte_size);

	//pinned directly
	int* h_temp;
	gpuErrchk(cudaMallocHost((int**)&h_temp, byte_size))

		int* d_results;

	gpuErrchk(cudaMalloc((void**)&d_results, byte_size));

	for(int i = 0; i < size; i++)
	{
		h_temp[i] = 5;
	}

	gpuErrchk(cudaMemcpy(d_results, h_temp, byte_size, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(h_temp, d_results, byte_size, cudaMemcpyDeviceToHost));

	return 0;

	// nvprof --print-gpu-trace
}
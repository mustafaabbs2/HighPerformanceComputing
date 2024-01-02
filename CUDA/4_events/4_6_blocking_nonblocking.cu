
//C headers
#include <stdio.h>
#include <stdlib.h>
//need these CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Contains self written helper functions
#include "../common/common.h"

__global__ void blocking_nonblocking_test1()
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if(gid == 0)
	{
		for(size_t i = 0; i < 10000; i++)
		{
			printf("kernel 1 \n");
		}
	}
}

int main(int argc, char** argv)
{
	int size = 1 << 15;

	cudaStream_t stm1, stm2, stm3;
	CHECK(cudaStreamCreateWithFlags(&stm1, cudaStreamNonBlocking));
	CHECK(cudaStreamCreate(&stm2));
	CHECK(cudaStreamCreateWithFlags(&stm3, cudaStreamNonBlocking));

	dim3 block(128);
	dim3 grid(size / block.x);

	blocking_nonblocking_test1<<<grid, block, 0, stm1>>>();
	blocking_nonblocking_test1<<<grid, block, 0, stm2>>>();
	blocking_nonblocking_test1<<<grid, block, 0, stm3>>>();

	CHECK(cudaStreamDestroy(stm1));
	CHECK(cudaStreamDestroy(stm2));
	CHECK(cudaStreamDestroy(stm3));
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaDeviceReset());
	return 0;
}
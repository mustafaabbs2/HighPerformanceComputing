#include <stdio.h>
#include <stdlib.h>
//need these CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Contains self written helper functions
#include "../common/common.h"

__global__ void kernel()
{
	printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, blockDim.x: %d, blockDim.y: %d, "
		   "blockDim.z: %d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d",
		   blockIdx.x,
		   blockIdx.y,
		   blockIdx.z,
		   blockDim.x,
		   blockDim.y,
		   blockDim.z,
		   gridDim.x,
		   gridDim.y,
		   gridDim.z);
}

int main(void)
{
	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4; //total 64 threads

	dim3 block(2, 2, 2); //block with 2 threads in each direction
	dim3 grid(nx / block.x, ny / block.y, nz / block.z); //two blocks in each direction

	kernel<<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void print_as_3d(int* aInput)
{
	int tid = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) +
			  threadIdx.x; // Z | Y | X

	int nThreadsInBlock = blockDim.x * blockDim.y * blockDim.z;
	int block_offset_x =
		blockIdx.x *
		nThreadsInBlock; //how to advance in a row major direction - add number of threads in a block in the x direction

	int nThreadsInLine = nThreadsInBlock * gridDim.y; //number of threads in a vertical line
	int block_offset_y = nThreadsInLine * blockIdx.y; //advance in the y direction

	int nThreadsInRect =
		nThreadsInLine * gridDim.x; // number of threads in one whole rectangular frame
	int block_offset_z = nThreadsInRect * blockIdx.z; //advance in the z direction

	int gid = block_offset_x + block_offset_y + block_offset_z + tid;
	printf("tid: %d, gid: %d, value: %d\n", threadIdx.x, gid, aInput[gid]);
}

int main()
{

	int nArrSize = 64; // X = 4, Y = 4, Z = 4
	int nByteSize = nArrSize * sizeof(int);

	int* h_aInput = nullptr;
	h_aInput = (int*)malloc(nByteSize);

	time_t t;
	srand((unsigned)time(&t));
	for(int i = 0; i < nArrSize; i++)
	{
		h_aInput[i] = (int)(rand() & 0xff);
	}

	int* d_aInput = nullptr;
	cudaMalloc((void**)&d_aInput, nByteSize);
	cudaMemcpy(d_aInput, h_aInput, nByteSize, cudaMemcpyHostToDevice);

	dim3 block(2, 2, 2);
	dim3 grid(2, 2, 2); // (4 / block.x, 4 / block.y, 4 / block.z)

	print_as_3d<<<grid, block>>>(d_aInput);
	cudaDeviceSynchronize();
	cudaFree(d_aInput);
	cudaDeviceReset();

	free(h_aInput);

	return 0;
}
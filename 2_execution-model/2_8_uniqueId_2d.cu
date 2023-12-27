#include <iostream>

__global__ void unique_gid_2d()
{

	int tid =
		blockDim.x * threadIdx.y + threadIdx.x; // number threads in a block in a row-major form

	int block_offset =
		blockIdx.x * blockDim.x; //block_offset is added when you move to the next block along a row

	int row_offset =
		gridDim.x * blockDim.x *
		blockIdx.y; //how many threads in a row of blocks - row offset is added when blockIdx.y > 0

	int gid = tid + block_offset + row_offset;

	printf("blockIdx.x: %d, blockIdx.y: %d, gid: %d", blockIdx.x, blockIdx.y, gid);
}

int main(void)
{

	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4; //total 64 threads

	dim3 block(2, 2, 2); //block with 2 threads in each direction
	dim3 grid(nx / block.x, ny / block.y, nz / block.z); //two blocks in each direction

	unique_gid_2d<<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
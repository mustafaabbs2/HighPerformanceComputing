#include <iostream>

__global__ void print_warp()
{

	int gid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x * gridDim.x * blockIdx.y;

	int warp_id = threadIdx.x / 32;

	int gbid = blockIdx.y * gridDim.x + blockIdx.x;

	printf(" tid.x : %d, gid : %d, warp_id : %d , gbid : %d", threadIdx.x, gid, warp_id, gbid);
}

int main(void)
{

	int nx, ny;
	nx = 16;
	ny = 16; //total 256 threads

	//dim3 block(8,8); //block with 8 threads in each direction
	//dim3 grid(nx/block.x, ny/block.y); //two blocks in each direction

	dim3 block(42);
	dim3 grid(2, 2);

	print_warp<<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
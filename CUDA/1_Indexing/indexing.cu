#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <vector>

__global__ void helloFromGPU()
{
	printf("Hello World from GPU! \n");
}

__global__ void add(int a, int b, int* c)
{
	*c = a + b;
}

__global__ void checkIndex()
{
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
		   "gridDim:(%d, %d, %d)\n",
		   threadIdx.x,
		   threadIdx.y,
		   threadIdx.z,
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

__global__ void checkIndex3D()
{
	printf("blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
		   "gridDim:(%d, %d, %d)\n",
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

__global__ void unique1D(int* input)
{

	int gid =
		blockIdx.x * blockDim.x + threadIdx.x; // number threads in a block in a row-major form

	printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
}

__global__ void unique2D()
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

__global__ void printWarpIndex()
{

	int gid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x * gridDim.x * blockIdx.y;

	int warp_id = threadIdx.x / 32;

	int gbid = blockIdx.y * gridDim.x + blockIdx.x;

	printf(" tid.x : %d, gid : %d, warp_id : %d , gbid : %d", threadIdx.x, gid, warp_id, gbid);
}

//Wrapper functions begin here //////////////

void helloFromGPU_()
{
	helloFromGPU<<<10, 1>>>();
}

void add_(int a, int b, int* c)
{
	int* dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	add<<<1, 1>>>(a, b, dev_c);
	cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_c);
}

void getDevice_()
{
	cudaDeviceProp prop;
	int count = 5;
	cudaGetDeviceCount(&count);

	cudaError_t rc, rd;

	int driver_version = 0, runtime_version = 0;

	rc = cudaDriverGetVersion(&driver_version);
	printf("Driver Version %d \n", driver_version);
	rd = cudaRuntimeGetVersion(&runtime_version);
	printf("Runtime Version %d \n", runtime_version);

	for(int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Total global memory %ld\n", prop.totalGlobalMem);
		printf("Total constant memory %ld\n", prop.totalConstMem);
		printf("Max memory pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared memory per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n",
			   prop.maxThreadsDim[0],
			   prop.maxThreadsDim[1],
			   prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n",
			   prop.maxGridSize[0],
			   prop.maxGridSize[1],
			   prop.maxGridSize[2]);
		printf("\n");
	}
}

void checkIndex_()
{
	int nElem = 6;
	dim3 block(3);
	dim3 grid((nElem + block.x - 1) / block.x);

	// check grid and block dimension from host side --> 2 1-D blocks of 3 threads each
	printf("\n From the host:\n");
	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

	// check grid and block dimension from device side
	printf("\n From the device:\n");
	checkIndex<<<grid, block>>>();
	cudaDeviceReset();
}

void checkIndex3D_()
{
	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4; //total 64 threads

	dim3 block(2, 2, 2); //block with 2 threads in each direction
	dim3 grid(nx / block.x, ny / block.y, nz / block.z); //two blocks in each direction

	checkIndex3D<<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
}

void unique1D_()
{

	int size = 32;

	std::vector<int> h_input(size);

	// Seed for random number generation
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	for(int i = 0; i < size; ++i)
	{
		h_input[i] = static_cast<int>(std::rand() & 0xff);
		//0xff = 11111111 -> masks all but the last 8 bits, so number is between 0 and 255
	}

	int* d_input;

	int byte_size = size * sizeof(int);

	cudaMalloc((void**)&d_input, byte_size);

	cudaMemcpy(d_input, h_input.data(), byte_size, cudaMemcpyHostToDevice);

	dim3 block(32);
	dim3 grid(1);

	unique1D<<<grid, block>>>(d_input);
	cudaDeviceSynchronize();
	cudaFree(d_input);

	cudaDeviceReset();
}

void unique2D_()
{

	int nx, ny, nz;
	nx = 4;
	ny = 4;
	nz = 4; //total 64 threads

	dim3 block(2, 2, 2); //block with 2 threads in each direction
	dim3 grid(nx / block.x, ny / block.y, nz / block.z); //two blocks in each direction

	unique2D<<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
}

void printWarpIndex_()
{

	int nx, ny;
	nx = 8;
	ny = 8; //64 threads in total

	//2 warps:
	dim3 block(64);
	dim3 grid(1);

	printWarpIndex<<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
}
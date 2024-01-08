#include <stdio.h>

__global__ void helloFromGPU()
{
	printf("Hello World from GPU! \n");
}

__global__ void add(int a, int b, int* c)
{
	*c = a + b;
}

void add_(int a, int b, int* c)
{
	int* dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	add<<<1, 1>>>(a, b, dev_c);
	cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_c);
}

void helloFromGPU_()
{
	helloFromGPU<<<10, 1>>>();
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
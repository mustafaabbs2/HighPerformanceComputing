#include "device_launch_parameters.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU! \n");
}

__global__ void addStreams(int* in, int* out, int size)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if(gid < size)
	{
		for(int i = 0; i < 25; i++)
		{
			out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
		}
	}
}

//Wrapper Functions:

void helloWorldEvents_()
{
	printf("Hello World from CPU! \n");

	float et;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Start timing
	cudaEventRecord(start);

	helloFromGPU<<<1, 10>>>();

	//Synchronize
	cudaDeviceSynchronize();

	//Stop
	cudaEventRecord(stop);

	//Sync events
	cudaEventSynchronize(stop);

	//Calculate et = elapsed time
	cudaEventElapsedTime(&et, start, stop);

	printf("The elapsed time is  %f milliseconds", et);

	cudaDeviceReset();
}

void helloWorldStreams_()
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	if(deviceProp.concurrentKernels == 0)
	{
		printf("> GPU does not support concurrent kernel execution \n");
		printf("kernel execution will be serialized \n");
	}

	cudaStream_t str1, str2, str3;

	cudaStreamCreate(&str1);
	cudaStreamCreate(&str2);
	cudaStreamCreate(&str3);

	helloFromGPU<<<1, 1, 0, str1>>>();
	helloFromGPU<<<1, 1, 0, str2>>>();
	helloFromGPU<<<1, 1, 0, str3>>>();

	cudaStreamDestroy(str1);
	cudaStreamDestroy(str2);
	cudaStreamDestroy(str3);

	cudaDeviceSynchronize();
	cudaDeviceReset();
}

void addStreams_()
{
	int size = 1 << 18;
	int byte_size = size * sizeof(int);

	//initialize host pointer
	int *h_in, *h_ref, *h_in2, *h_ref2;

	cudaMallocHost((void**)&h_in, byte_size);
	cudaMallocHost((void**)&h_ref, byte_size);
	cudaMallocHost((void**)&h_in2, byte_size);
	cudaMallocHost((void**)&h_ref2, byte_size);

	//allocate device pointers
	int *d_in, *d_out, *d_in2, *d_out2;
	cudaMalloc((void**)&d_in, byte_size);
	cudaMalloc((void**)&d_out, byte_size);
	cudaMalloc((void**)&d_in2, byte_size);
	cudaMalloc((void**)&d_out2, byte_size);

	cudaStream_t str, str2;
	cudaStreamCreate(&str);
	cudaStreamCreate(&str2);

	//kernel launch
	dim3 block(128);
	dim3 grid(size / block.x);

	//transfer data from host to device
	cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice, str);
	addStreams<<<grid, block, 0, str>>>(d_in, d_out, size);
	cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost, str);

	cudaMemcpyAsync(d_in2, h_in2, byte_size, cudaMemcpyHostToDevice, str2);
	addStreams<<<grid, block, 0, str2>>>(d_in2, d_out2, size);
	cudaMemcpyAsync(h_ref2, d_out2, byte_size, cudaMemcpyDeviceToHost, str2);

	cudaStreamSynchronize(str);
	cudaStreamDestroy(str);

	cudaStreamSynchronize(str2);
	cudaStreamDestroy(str2);

	cudaDeviceReset();
}

//C headers
#include <stdio.h>
#include <stdlib.h>
//need these CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Contains self written helper functions
#include "../common/common.h"

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData;

__global__ void checkGlobalVariable()
{
	// display the original value
	printf("Device: the value of the global variable is %f\n", devData);

	// alter the value
	devData += 2.0f;
}

int main(void)
{
	// initialize the global variable
	float value = 3.14f;
	CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float))); //devData is passed as a symbol
	printf("Host:   copied %f to the global variable\n", value);

	// invoke the kernel
	checkGlobalVariable<<<1, 1>>>();

	// copy the global variable back to the host
	CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
	printf("Host:   the value changed by the kernel to %f\n", value);

	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}

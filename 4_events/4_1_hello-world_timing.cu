#include <iostream>

__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU! \n");
}

int main()
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
	return 0;
}
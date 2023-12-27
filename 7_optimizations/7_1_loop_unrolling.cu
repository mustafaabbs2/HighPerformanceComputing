#include <cuda_runtime.h>
#include <iostream>

__global__ void unrolled_kernel(int* data, int size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
	{
		data[idx] = data[idx] * 2;
		data[idx + 1] = data[idx + 1] * 2;
		data[idx + 2] = data[idx + 2] * 2;
		data[idx + 3] = data[idx + 3] * 2;
	}
}

int main()
{
	int size = 64;
	int *data, *d_data;
	cudaMalloc(&d_data, size * sizeof(int));
	data = new int[size];

	for(int i = 0; i < size; i++)
	{
		data[i] = i;
	}

	cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

	unrolled_kernel<<<1, size / 4>>>(d_data, size);
	cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < size; i++)
	{
		std::cout << data[i] << " ";
	}

	cudaFree(d_data);
	delete[] data;

	return 0;
}

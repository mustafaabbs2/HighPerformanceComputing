#include "utilityKernels.h"

__global__ void copy_array(float* u, float* u_prev, int N, int BSZ)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y * BSZ * N + blockIdx.x * BSZ + j * N + i;
	if(I >= N * N)
	{
		return;
	}
	u_prev[I] = u[I];
}

void copy_array_(float* u_d, float* u_prev_d, int N, int BSZ, dim3 dimGrid, dim3 dimBlock)
{
	copy_array<<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, BSZ);
}
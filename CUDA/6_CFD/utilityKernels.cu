#include "utilityKernels.cuh"

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

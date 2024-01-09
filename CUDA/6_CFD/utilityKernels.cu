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

__global__ void copy_kernel(float* u, float* u_prev, int N, int BSZ, int N_max)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int x = i + blockIdx.x * BSZ;
	int y = j + blockIdx.y * BSZ;
	int I = x + y * N_max;

	float value = tex2D(tex_u, x, y);

	u_prev[I] = value;
}
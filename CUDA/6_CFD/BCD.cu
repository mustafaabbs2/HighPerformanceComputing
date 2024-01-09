/*** Calculating a derivative with CD ***/
#include <cmath>
#include <fstream>
#include <iostream>

#include "BCD.h"
#include "utilityKernels.cuh"

//CPU Kernel
void updateCPU(
	std::vector<float>& u, std::vector<float>& u_prev, size_t N, float h, float dt, float alpha)
{
	int I;
	for(int j = 0; j < N; j++)
	{
		for(int i = 0; i < N; i++)
		{
			I = j * N + i; //converting to row major form
			u_prev[I] = u[I];
		}
	}

	for(int j = 1; j < N - 1; j++)
	{
		for(int i = 1; i < N - 1; i++) //for the interior nodes
		{
			I = j * N + i;
			u[I] = u_prev[I] + alpha * dt / (h * h) *
								   (u_prev[I + 1] + u_prev[I - 1] + u_prev[I + N] + u_prev[I - N] -
									4 * u_prev[I]);
		}
	}
}

// GPU kernel
__global__ void update(float* u, float* u_prev, int N, float h, float dt, float alpha, int BSZ)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int I = blockIdx.y * BSZ * N + blockIdx.x * BSZ + j * N + i;

	if(I >= N * N)
	{
		return;
	}

	if((I > N) && (I < N * N - 1 - N) && (I % N != 0) && (I % N != N - 1))
	{
		u[I] = u_prev[I] +
			   alpha * dt / (h * h) *
				   (u_prev[I + 1] + u_prev[I - 1] + u_prev[I + N] + u_prev[I - N] - 4 * u_prev[I]);
	}
}

void updateGPU(float* u_d,
			   float* u_prev_d,
			   int N,
			   float h,
			   float dt,
			   float alpha,
			   int BSZ,
			   dim3 dimGrid,
			   dim3 dimBlock)

{
	update<<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, h, dt, alpha, BSZ);
}

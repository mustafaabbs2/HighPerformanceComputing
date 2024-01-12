#include "BCD.h"
#include "utilityKernels.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

void writeFile(std::string filename,
			   const size_t N,
			   std::vector<float>& x,
			   std::vector<float>& y,
			   std::vector<float>& u)
{
	int I;
	std::ofstream filehandle(filename);
	for(auto j = 0; j < N; j++)
		for(auto i = 0; i < N; i++)
		{
			I = N * j + i;
			filehandle << x[I] << " " << y[I] << " " << u[I] << std::endl;
		}

	filehandle.close();
}

static float
setupInitialCondition(size_t N, std::vector<float>& x, std::vector<float>& y, std::vector<float>& u)
{
	// Generate mesh and intial condition
	float xmin = 0.0f;
	float xmax = 3.5f;
	float ymin = 0.0f;

	float h = (xmax - xmin) / (N - 1);
	size_t I;

	for(size_t j = 0; j < N; j++)
	{
		for(size_t i = 0; i < N; i++)
		{
			I = N * j + i;
			x[I] = xmin + h * i;
			y[I] = ymin + h * j;
			u[I] = 0.0f;
			if((i == 0) || (j == 0))
			{
				u[I] = 200.0f;
			}
		}
	}

	return h;
}

int main()
{
	auto N = 128;
	bool cpu = false;

	auto dt = 0.00001;
	auto alpha = 0.645;
	auto time = 0.4;
	auto steps = ceil(time / dt);

	auto x = std::make_unique<std::vector<float>>(N * N);
	auto y = std::make_unique<std::vector<float>>(N * N);
	auto u = std::make_unique<std::vector<float>>(N * N);
	auto u_prev = std::make_unique<std::vector<float>>(N * N);

	auto h = setupInitialCondition(N, *x, *y, *u);

	if(cpu)
	{
		for(auto t = 0; t < steps; t++)
		{
			update(*u, *u_prev, N, h, dt, alpha);
		}
	}
	else
	{
		int BLOCKSIZE = 16;

		// Allocate in GPU
		float *u_d, *u_prev_d;

		cudaMalloc((void**)&u_d, N * N * sizeof(float));
		cudaMalloc((void**)&u_prev_d, N * N * sizeof(float));

		// Copy to GPU
		cudaMemcpy(u_d, u.get(), N * N * sizeof(float), cudaMemcpyHostToDevice);

		// Loop
		dim3 dimGrid(int((N - 0.5) / BLOCKSIZE) + 1, int((N - 0.5) / BLOCKSIZE) + 1);
		dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

		for(int t = 0; t < steps; t++)
		{
			copy_array_(u_d, u_prev_d, N, BLOCKSIZE, dimGrid, dimBlock);
			update_(u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE, dimGrid, dimBlock);
		}

		//Synchronize
		cudaDeviceSynchronize();

		// Copy result back to host
		cudaMemcpy(u.get(), u_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

		// Free device
		cudaFree(u_d);
		cudaFree(u_prev_d);
	}

	writeFile("test.txt", N, *x, *y, *u);
}
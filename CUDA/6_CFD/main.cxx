#include "BCD.cuh"
#include <fstream>
#include <iostream>

void writeFile(std::string filename,
			   const size_t N,
			   std::vector<float>& x,
			   std::vector<float>& y,
			   std::vector<float>& u)
{
	std::ofstream variable(filename);
	for(size_t j = 0; j < N; j++)
	{
		for(int i = 0; i < N; i++)
		{
			I = N * j + i;
			variable << x[I] << " " << y[I] << " " << u[I] << std::endl;
		}
	}

	variable.close();
}

static void
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
}

// void setupGPU()
// {
// 	int BLOCKSIZE = 16;

// 	float et;
// 	cudaEvent_t start, stop;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);

// 	cudaSetDevice(2);

// 	// Allocate in GPU
// 	float *u_d, *u_prev_d;

// 	cudaMalloc((void**)&u_d, N * N * sizeof(float));
// 	cudaMalloc((void**)&u_prev_d, N * N * sizeof(float));

// 	// Copy to GPU
// 	cudaMemcpy(u_d, u, N * N * sizeof(float), cudaMemcpyHostToDevice);

// 	// Loop
// 	dim3 dimGrid(int((N - 0.5) / BLOCKSIZE) + 1, int((N - 0.5) / BLOCKSIZE) + 1);
// 	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

// 	//Start timing
// 	cudaEventRecord(start);

// 	for(int t = 0; t < steps; t++)
// 	{
// 		copy_array<<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, BLOCKSIZE);
// 		update<<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE);
// 	}

// 	//Synchronize
// 	cudaDeviceSynchronize();

// 	//Stop
// 	cudaEventRecord(stop);

// 	//Sync events
// 	cudaEventSynchronize(stop);

// 	//Calculate et = elapsed time
// 	cudaEventElapsedTime(&et, start, stop);

// 	//Calculate et = elapsed time
// 	cudaEventElapsedTime(&et, start, stop);

// 	printf("The elapsed time is  %f milliseconds", et);

// 	// Copy result back to host
// 	cudaMemcpy(u, u_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

// 	// Free device
// 	cudaFree(u_d);
// 	cudaFree(u_prev_d);
// }

int main()
{

	auto N = 128;

	auto x = std::make_unique<std::vector<float>>(N * N);
	auto y = std::make_unique<std::vector<float>>(N * N);
	auto u = std::make_unique<std::vector<float>>(N * N);

	setupInitialCondition(N, *x, *y, *u);

	for(int t = 0; t < steps; t++)
	{
		update(u, u_prev, N, h, dt, alpha);
	}

	writeFile("test.txt", N, *x, *y, *u);
}
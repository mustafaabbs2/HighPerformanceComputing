/*** Calculating a derivative with CD ***/
#include <cmath>
#include <fstream>
#include <iostream>

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

int main()
{
	// Allocate in CPU
	int N = 128;
	int BLOCKSIZE = 16;
	//Timing vars
	float et;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaSetDevice(2);

	float xmin = 0.0f;
	float xmax = 3.5f;
	float ymin = 0.0f;
	//float ymax 	= 2.0f;
	float h = (xmax - xmin) / (N - 1);
	float dt = 0.00001f;
	float alpha = 0.645f;
	float time = 0.4f;

	int steps = ceil(time / dt);
	int I;

	float* x = new float[N * N];
	float* y = new float[N * N];
	float* u = new float[N * N];
	float* u_prev = new float[N * N];

	// Generate mesh and intial condition
	for(int j = 0; j < N; j++)
	{
		for(int i = 0; i < N; i++)
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

	// Allocate in GPU
	float *u_d, *u_prev_d;

	cudaMalloc((void**)&u_d, N * N * sizeof(float));
	cudaMalloc((void**)&u_prev_d, N * N * sizeof(float));

	// Copy to GPU
	cudaMemcpy(u_d, u, N * N * sizeof(float), cudaMemcpyHostToDevice);

	// Loop
	dim3 dimGrid(int((N - 0.5) / BLOCKSIZE) + 1, int((N - 0.5) / BLOCKSIZE) + 1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

	//Start timing
	cudaEventRecord(start);

	for(int t = 0; t < steps; t++)
	{
		copy_array<<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, BLOCKSIZE);
		update<<<dimGrid, dimBlock>>>(u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE);
	}

	//Synchronize
	cudaDeviceSynchronize();

	//Stop
	cudaEventRecord(stop);

	//Sync events
	cudaEventSynchronize(stop);

	//Calculate et = elapsed time
	cudaEventElapsedTime(&et, start, stop);

	//Calculate et = elapsed time
	cudaEventElapsedTime(&et, start, stop);

	printf("The elapsed time is  %f milliseconds", et);

	// Copy result back to host
	cudaMemcpy(u, u_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	std::ofstream temperature("temperature_global.txt");
	for(int j = 0; j < N; j++)
	{
		for(int i = 0; i < N; i++)
		{
			I = N * j + i;
			temperature << x[I] << " " << y[I] << " " << u[I] << std::endl;
		}
	}

	temperature.close();

	// Free device
	cudaFree(u_d);
	cudaFree(u_prev_d);
}

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

//helper functions
#define SHMEM_SIZE 256

void init_matrix(int* m, int N)
{
	int i, j;
	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++)
			m[i * N + j] = rand() % 10;
}

#define CHECK(call)                                                                                \
	{                                                                                              \
		const cudaError_t error = call;                                                            \
		if(error != cudaSuccess)                                                                   \
		{                                                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                                 \
			fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));           \
			exit(1);                                                                               \
		}                                                                                          \
	}

//device functions
__global__ void sumArray(int* a, int* b, int* c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
}

__global__ void sumReduction(int* v, int* v_r)
{
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	for(int s = 1; s < blockDim.x; s *= 2)
	{

		if(threadIdx.x % (2 * s) == 0)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
	{
		v_r[blockIdx.x] = partial_sum[0];
	}
}

__global__ void sumReductionNoDivergence(int* v, int* v_r)
{
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	for(int s = 1; s < blockDim.x; s *= 2)
	{

		// This modulo operation is bad!! - there is an if condition causing warp divergence
		// if (threadIdx.x % (2 * s) == 0) {
		// 	partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		// }
		// instead .. do..

		int index = 2 * s * threadIdx.x;
		// when s = 1, for threadIdx.x = 0, index = 0, the first thread adds partial_sum[0] to partial_sum[1]
		// when s = 1, for threadIdx.x = 1, index = 2, the second thread adds partial_sum[2] to partial_sum[3]
		//.. and so on

		if(index < blockDim.x)
		{
			partial_sum[index] += partial_sum[index + s];
		}

		__syncthreads();
	}

	if(threadIdx.x == 0)
	{
		v_r[blockIdx.x] = partial_sum[0];
	}
}

__global__ void sumReductionNoBankConflicts(int* v, int* v_r)
{
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for(int s = blockDim.x / 2; s > 0; s >>= 1)
	{ //s>>=1 means s/2
		// Each thread does work unless it is further than the stride
		if(threadIdx.x < s)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if(threadIdx.x == 0)
	{
		v_r[blockIdx.x] = partial_sum[0];
	}
}

__global__ void sumReductionReduceThreads(int* v, int* v_r)
{
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for(int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		// Each thread does work unless it is further than the stride
		if(threadIdx.x < s)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if(threadIdx.x == 0)
	{
		v_r[blockIdx.x] = partial_sum[0];
	}
}

__device__ void warpReduce(volatile int* shmem_ptr, int t)
{
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

__global__ void sumReductionUnrollLoops(int* v, int* v_r)
{
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	// int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	for(int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		// Each thread does work unless it is further than the stride
		if(threadIdx.x < s)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if(threadIdx.x < 32)
	{
		warpReduce(partial_sum, threadIdx.x);
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if(threadIdx.x == 0)
	{
		v_r[blockIdx.x] = partial_sum[0];
	}
}

__global__ void matrixMultiply(int* a, int* b, int* c, int N)
{
	int row = blockIdx.y * blockDim.y +
			  threadIdx.y; // threadIdx.y is the local row index and add it to the offset
	int col = blockIdx.x * blockDim.x +
			  threadIdx.x; //threadIdx.x is the local column index and add it to the offset

	if(row < N && col < N)
	{
		int sum = 0;
		for(int i = 0; i < N; i++)
		{
			sum += a[row * N + i] * b[i * N + col];
		}
		c[row * N + col] = sum;
	}
}

//wrapper functions begin here
void sumArray_()
{

	int size = 10;
	int byte_size = size * sizeof(int);

	int *h_a, *h_b, *h_c;
	h_a = (int*)malloc(byte_size);
	h_b = (int*)malloc(byte_size);
	h_c = (int*)malloc(byte_size);

	for(int i = 0; i < size; i++)
	{
		h_a[i] = i + 1;
		h_b[i] = i + 1;
		h_c[i] = 0;
	}

	//allocating memory on the device
	int *d_a, *d_b, *d_c;
	CHECK(cudaMalloc((void**)&d_a, byte_size));
	CHECK(cudaMalloc((void**)&d_b, byte_size));
	CHECK(cudaMalloc((void**)&d_c, byte_size));

	CHECK(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_c, h_c, byte_size, cudaMemcpyHostToDevice));

	dim3 block(64); //block with 64 threads in each direction
	dim3 grid(2); //two blocks

	sumArray<<<grid, block>>>(d_a, d_b, d_c, size);
	cudaDeviceSynchronize();

	CHECK(cudaMemcpy(h_c, d_c, byte_size, cudaMemcpyDeviceToHost));

	for(int i = 0; i < size; i++)
	{
		printf("%d ", h_c[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	cudaDeviceReset();
}

//reduce to the number of thread blocks first, and then do a second reduction
void sumReduction_()
{
	// Vector size
	int N = 1 << 16;
	size_t bytes = N * sizeof(int);

	// Host data
	std::vector<int> h_v(N);
	std::vector<int> h_v_r(N);

	// Initialize the input data
	std::generate(std::begin(h_v), std::end(h_v), []() { return rand() % 10; });

	// Allocate device memory
	int *d_v, *d_v_r;
	CHECK(cudaMalloc(&d_v, bytes));
	CHECK(cudaMalloc(&d_v_r, bytes));

	// Copy to device
	CHECK(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));

	// TB Size
	const int TB_SIZE = 256;

	int GRID_SIZE = N / TB_SIZE;

	std::cout << "Sum Reduction with Partial Sums" << std::endl;
	auto start_time = std::chrono::high_resolution_clock::now();

	sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
	sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r);

	auto end_time = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V0: " << duration.count() << " microseconds" << std::endl;

	// Copy to host;
	CHECK(cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost));

	assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

	//V1///////////////////////////////////

	std::cout << "Sum Reduction with No Divergence" << std::endl;
	start_time = std::chrono::high_resolution_clock::now();

	sumReductionNoDivergence<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
	sumReductionNoDivergence<<<1, TB_SIZE>>>(d_v_r, d_v_r);

	end_time = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V1: " << duration.count() << " microseconds" << std::endl;

	// Copy to host;
	CHECK(cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost));

	assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

	//V2///////////////////////////////////
	std::cout << "Sum Reduction with No Bank Conflicts" << std::endl;
	start_time = std::chrono::high_resolution_clock::now();

	sumReductionNoBankConflicts<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
	sumReductionNoBankConflicts<<<1, TB_SIZE>>>(d_v_r, d_v_r);

	end_time = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V2: " << duration.count() << " microseconds" << std::endl;

	// Copy to host;
	CHECK(cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost));

	assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

	//V3///////////////////////////////////
	std::cout << "Sum Reduction with Reduced Threads" << std::endl;
	start_time = std::chrono::high_resolution_clock::now();

	sumReductionReduceThreads<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
	sumReductionReduceThreads<<<1, TB_SIZE>>>(d_v_r, d_v_r);

	end_time = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V3: " << duration.count() << " microseconds" << std::endl;

	// Copy to host;
	CHECK(cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost));

	assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

	//V4///////////////////////////////////
	std::cout << "Sum Reduction with Unrolled Loops" << std::endl;
	start_time = std::chrono::high_resolution_clock::now();

	sumReductionUnrollLoops<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
	sumReductionUnrollLoops<<<1, TB_SIZE>>>(d_v_r, d_v_r);

	end_time = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V4: " << duration.count() << " microseconds" << std::endl;

	// Copy to host;
	CHECK(cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost));

	assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));
}

void matrixMultiply_()
{
	// set our problem size for a square matrix
	int N = 1 << 10; // 2^10 = 1024
	size_t bytes = N * N * sizeof(int);

	//allocate memory for the matrices
	int *a, *b, *c;
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// initialize the matrices

	init_matrix(a, N);
	init_matrix(b, N);

	//Set up grid dimensions
	int threads = 16;
	int blocks = (N + threads - 1) / threads;

	//Set up the kernel launch configuration
	dim3 block(threads, threads);
	dim3 grid(blocks, blocks);

	std::cout << "Naive Matrix Multiplication" << std::endl;
	auto start_time = std::chrono::high_resolution_clock::now();

	matrixMultiply<<<grid, block>>>(a, b, c, N);

	auto end_time = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V0: " << duration.count() << " microseconds" << std::endl;

	cudaDeviceSynchronize();
}
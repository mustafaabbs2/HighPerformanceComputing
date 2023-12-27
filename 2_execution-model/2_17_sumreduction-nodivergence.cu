#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::vector;

#define SHMEM_SIZE 256

__global__ void sumReduction(int* v, int* v_r)
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

int main()
{
	// Vector size
	int N = 1 << 16;
	size_t bytes = N * sizeof(int);

	// Host data
	vector<int> h_v(N);
	vector<int> h_v_r(N);

	// Initialize the input data
	generate(begin(h_v), end(h_v), []() { return rand() % 10; });

	// Allocate device memory
	int *d_v, *d_v_r;
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	// Copy to device
	cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);

	// TB Size
	const int TB_SIZE = 256;

	int GRID_SIZE = N / TB_SIZE;

	sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
	sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r);

	// Copy to host;
	cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

	// Print the result
	assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

	cout << "COMPLETED SUCCESSFULLY\n";

	return 0;
	// nvcc -o 2_17_sumreduction-nodivergence 2_17_sumreduction-nodivergence.cu
}
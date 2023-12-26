// This matrix computes matrix multiplication C = A * B on the GPU using CUDA

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>



__global__ void matrixMul(int *a, int *b, int *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // threadIdx.y is the local row index and add it to the offset
    int col = blockIdx.x * blockDim.x + threadIdx.x; //threadIdx.x is the local column index and add it to the offset

    if (row < N && col < N)
    {
        int sum = 0;
        for (int i = 0; i < N; i++)
        {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }


}

//initialize a square matrix with random values

void init_matrix(int *m, int N)
{
    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            m[i * N + j] = rand() % 10;

}


//verify the result on CPU

void verify_result(int *a, int *b, int *c, int N)
{
    int i, j, k, tmp;
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            tmp = 0;
            for (k =0; k < N; k++)
                {
                    tmp += a[i * N + k] * b[k * N + j];
                }
        }
        // check each result
        assert(tmp - c[i * N + j] <1e-3);
    }    
}


int main()
{
// set our problem size for a square matrix
int N = 1 << 10;   // 2^10 = 1024
size_t bytes =  N * N * sizeof(int);


//allocate memory for the matrices

int *a, *b, *c;
cudaMallocManaged(&a, bytes);
cudaMallocManaged(&b, bytes);
cudaMallocManaged(&c, bytes);

// initialize the matrices

init_matrix(a,N);
init_matrix(b,N);

//Set up grid dimensions
int threads = 16;
int blocks = (N + threads - 1) / threads;

//Set up the kernel launch configuration
dim3 block(threads, threads);
dim3 grid(blocks, blocks);

// Launch the kernel

matrixMul<<<grid, block>>>(a, b, c, N);
cudaDeviceSynchronize();//or call cudamemcopy - which is a synchronous call


// Verify the result
verify_result(a, b, c, N);

    return 0;
}
#include <iostream>

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include <thrust/device_ptr.h>

#define BSZ 2

__global__ void csr_matvec(float *elem, int *row_off, int *col, float *b, float *c, int N_side)
{
	// index i runs through the rows of the matrix
	int i = blockIdx.x * BSZ + threadIdx.x;

	if (i>=N_side) {return;}
	
	int row_begin 	= row_off[i];
	int row_end	= row_off[i+1];

	float sum = 0.0f;
	int column;
	for (int j=row_begin; j<row_end; j++)
	{	column = col[j];
		sum += elem[j]*b[column];
	}
	c[i] = sum;
	
}


int main()
{	
	int N = 4;
	int N_side = N*N;
	//int N_el = 5*N_side-4*N;

	// Allocate and create matrix (the poisson5pt matrix has 5*N*N-4*N elements)
	cusp::csr_matrix <int, float, cusp::device_memory> A;
	cusp::gallery::poisson5pt(A, N, N);

	// Allocate and create vector
	cusp::array1d <float, cusp::device_memory> b(N_side, 1.0f);

	//cusp::array1d <float, cusp::device_memory> c(N_side);
	float *c_d;
	cudaMalloc( (void**) &c_d, N_side*sizeof(float));
	thrust::device_ptr<float> c_ptr(c_d);

	// Recover array pointers
	float *elements = thrust::raw_pointer_cast(&A.values[0]);
	int   *rows_off = thrust::raw_pointer_cast(&A.row_offsets[0]);
	int   *columns 	= thrust::raw_pointer_cast(&A.column_indices[0]);
	float *b_d	= thrust::raw_pointer_cast(&b[0]);
	//float *c_d	= thrust::raw_pointer_cast(&c[0]);

	int dimGrid (int((N_side-0.5)/BSZ) + 1);
	int dimBlock (BSZ);

	csr_matvec <<<dimGrid, dimBlock>>> (elements, rows_off, columns, b_d, c_d, N_side);

	cusp::array1d<float, cusp::device_memory> c(c_ptr, c_ptr+N_side);	

	cusp::print(c);

	return 0;

}

#include <iostream>

#include <cusp/array1d.h>
#include <cusp/ell_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <thrust/device_ptr.h>

#define BSZ 4

__global__ void ell_matvec(float *elem, int *col, float *b, float *c, int num_entries, int N_side)
{
	// write your kernel here!	
}


int main()
{	
	int N = 4;
	int N_side = N*N;

	// Allocate and create matrix
	cusp::ell_matrix <int, float, cusp::device_memory> A;
	cusp::gallery::poisson5pt(A, N, N);

	int ell_col = A.values.num_cols;
	int ell_row = A.values.num_rows;

	cusp::array1d <float, cusp::device_memory> A_val(ell_row*ell_col);
	cusp::array1d <float, cusp::device_memory> A_col(ell_row*ell_col);

	// Allocate and create vector
	cusp::array1d <float, cusp::device_memory> b(N_side, 1.0f);

        float *c_d;
        cudaMalloc( (void**) &c_d, N_side*sizeof(float));
        thrust::device_ptr<float> c_ptr(c_d);

	// Recover array pointers
	// array2d from ell comes in column major, need to transpose
	// before flatten as raw pointer cast of array2d assumes row major
	cusp::array2d <float, cusp::device_memory> At_val;
	cusp::transpose(A.values, At_val);
	cusp::array2d <int, cusp::device_memory> At_col;
	cusp::transpose(A.column_indices, At_col);

	float *val_d = thrust::raw_pointer_cast(&At_val(0,0));
	int   *col_d = thrust::raw_pointer_cast(&At_col(0,0));
	float *b_d   = thrust::raw_pointer_cast(&b[0]);

	//thrust::device_ptr<float> val_ptr(val_d);

	int dimGrid (int((N_side-0.5)/BSZ) + 1);
	int dimBlock (BSZ);

	ell_matvec <<<dimGrid, dimBlock>>> (val_d, col_d, b_d, c_d, ell_col, N_side);

	cusp::array1d<float, cusp::device_memory> c(c_ptr, c_ptr+N_side);

	cusp::print(c);

	return 0;

}

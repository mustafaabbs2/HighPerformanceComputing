#include <iostream>
#include <fstream>
#include <cmath>

#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/gallery/poisson.h>
#include <cusp/linear_operator.h>
#include <cusp/krylov/cg.h>
#include <cusp/print.h>
#include <cusp/blas.h>
#include <thrust/device_ptr.h>
#include <cuda.h>

#define BSZ (16)

void checkErrors(char *label)
{
// we need to synchronise first to catch errors due to
// asynchroneous operations that would otherwise
// potentially go unnoticed
cudaError_t err;
err = cudaThreadSynchronize();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
err = cudaGetLastError();
if (err != cudaSuccess)
{
char *e = (char*) cudaGetErrorString(err);
fprintf(stderr, "CUDA Error: %s (at %s)\n", e, label);
}
}

// GPU kernel
__global__ void compute_RHS (float *RHS, float *u, int N, float h, float dt, float alpha, float T_bound)
{	/**** Write your kernel here ****/
	/**** Remember to add the boundary condition to the right hand side ***/
}

template <typename MatrixType>
void generateMatrix (MatrixType &A, float alpha, float dt, float h, int N)
{	/**** Generate your matrix here ****/	
	/**** Remember M = I + alpha*dt/(2*h*h)*poisson5pt ****/
}

__global__ void copy_u_gpu (float *u_full, float *u, int N)
{	/**** Copy to large array here ****/
}


int main()
{
	// Allocate in CPU
	int N = 128;
	int BLOCKSIZE = BSZ;

	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	//float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.0001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;
	float T_bound 	= 200.0f;

	int steps = ceil(time/dt);
	int I;

	cusp::array1d<float, cusp::host_memory> x(N*N); 
	cusp::array1d<float, cusp::host_memory> y(N*N); 
	cusp::array1d<float, cusp::host_memory> u(N*N); 

	float *u_h = new float [N*N]; 

	// Generate mesh and intial condition
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i==0) || (j==0)) 
				{u[I] = T_bound;}
		}
	}

	// Allocate and copy in GPU
	cusp::array1d<float, cusp::device_memory> u_d_cusp_full = u; 
	float *u_d_full = thrust::raw_pointer_cast(&u_d_cusp_full[0]);
	cusp::array1d<float, cusp::device_memory> RHS_cusp((N-2)*(N-2), 0.0); 
	float *RHS = thrust::raw_pointer_cast(&RHS_cusp[0]);
	cusp::array1d <float, cusp::device_memory> u_d_cusp ((N-2)*(N-2), 0.0);
	float *u_d = thrust::raw_pointer_cast(&u_d_cusp[0]);

	//Generate matrix
	cusp::coo_matrix <int, float, cusp::device_memory> A ((N-2)*(N-2), (N-2)*(N-2), 5*(N-2)*(N-2)-4*(N-2));
	cusp::identity_operator<float, cusp::device_memory> ID(A.num_rows, A.num_rows);

	generateMatrix(A, alpha, dt, h, N);

	// Loop 
	dim3 (); // Total number of blocks?
	dim3 (); // Numer of threads per block?
	for (int t=0; t<steps; t++)
	{	
		// Generate RHS
		compute_RHS <<<dimGrid, dimBlock>>> (RHS, u_d_full, N, h, dt, alpha, T_bound);
		
		// Solve system
		cusp::default_monitor <float> monitor(RHS_cusp, 1000);
		cusp::krylov::cg(A, u_d_cusp, RHS_cusp, monitor, ID);

		//cusp::print(RHS_cusp);
		//return 0;
		copy_u_gpu <<<dimGrid, dimBlock>>> (u_d_full, u_d, N);
	}
	checkErrors("compute_RHS");
	
	// Copy result back to host
	cudaMemcpy(u_h, u_d_full, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	std::ofstream temperature("temperature_implicit.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
//			std::cout<<u[I]<<"\t";
			temperature<<x[I]<<"\t"<<y[I]<<"\t"<<u_h[I]<<std::endl;
		}
		temperature<<"\n";
//		std::cout<<std::endl;
	}

	temperature.close();
}

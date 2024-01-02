/*** Calculating a derivative with CD ***/
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
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int I_0 = by*N*BSZ + bx*BSZ; //Index of the first element of each block including halo
	int Index = by*BSZ*N + bx*BSZ + (j+1)*N + i+1; // Global index without halo
	int Index2 = by*BSZ*(N-2) + bx*BSZ + j*(N-2) + i; 

	int G_I = bx*BSZ+i; // Global I index
	int G_J = by*BSZ+j; // Global J index
	

	__shared__ float u_prev_sh[BSZ+2][BSZ+2];

	// Local indices inside block including halo
	int ii = j*BSZ + i;
	int I = ii%(BSZ+2); 
	int J = ii/(BSZ+2); 
	
	int I_n = I_0 + J*N + I; //General index

	// More data than threads-> Load to shared memory in two steps  
	// First step
	u_prev_sh[I][J] = u[I_n];

	// Second step
	int ii2 = BSZ*BSZ + j*BSZ + i;
	int I2  = ii2%(BSZ+2);
	int J2  = ii2/(BSZ+2);

	int I_n2 = I_0 + J2*N + I2; //General index

	if ( (I2<(BSZ+2)) && (J2<(BSZ+2)) && (ii2 < N*N) )
		u_prev_sh[I2][J2] = u[I_n2]; 

	__syncthreads();

	//if (Index>=N*N-1){return;}	
	if ( (G_J>N-3) || (G_I>N-3)){return;}	

	bool bound_check = ((Index>N) && (Index< N*N-1-N) && (Index%N!=0) && (Index%N!=N-1)); 
	//bool bound_check = ( (Index< N*N-1-N) && (Index%N!=N-1)); 

	// if not on boundary do 
	if (bound_check)
	{	RHS[Index2] = u_prev_sh[i+1][j+1] + alpha*dt/(2*h*h) * (u_prev_sh[i+2][j+1] + u_prev_sh[i][j+1] + u_prev_sh[i+1][j+2] + u_prev_sh[i+1][j] - 4*u_prev_sh[i+1][j+1]);
	}

	// Apply BCs
	if (G_I == 0) 
	{	RHS[Index2] += T_bound*alpha*dt/(2*h*h);	
	}
	if (G_J == 0) 
	{	RHS[Index2] += T_bound*alpha*dt/(2*h*h);	
	}

	// Boundary conditions are automatically imposed
	// as we don't touch boundaries
}

template <typename MatrixType>
void generateMatrix (MatrixType &A, float alpha, float dt, float h, int N)
{	
	// Matrix will be M = I + alpha*dt/(2*h*h)*poisson5pt
	cusp::coo_matrix<int, float, cusp::device_memory> M;
	cusp::gallery::poisson5pt(M, N-2, N-2);

	// Create identity matrix
	cusp::coo_matrix<int, float, cusp::device_memory> I((N-2)*(N-2), (N-2)*(N-2), (N-2)*(N-2));
	for (int i=0; i<(N-2)*(N-2); i++)
	{	I.row_indices[i] = i;
		I.column_indices[i] = i;
		I.values[i] = 1.0f;
	}

	cusp::array1d<float, cusp::device_memory> M_aux(5*(N-2)*(N-2)-4*(N-2), 0.);
	
	cusp::blas::axpy(M.values, M_aux, alpha*dt/(2*h*h));
	
	M.values = M_aux;

	cusp::add(I, M, A);
}

__global__ void copy_u_gpu (float *u_full, float *u, int N)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int I_large = N*BSZ*by + N*(j+1) + BSZ*bx + i+1;  // Large data set index
	int I_small = (N-2)*BSZ*by + (N-2)*j + BSZ*bx + i; // Small data set index

	int G_I = bx*BSZ+i+1; // Global I index
	int G_J = by*BSZ+j+1; // Global J index
	if ( (G_J>=N-1) || (G_I>=N-1)){return;}	

	u_full[I_large] = u[I_small];

	// u already has the BCs untouched, no need to apply them explicitly
}


int main()
{
	// Allocate in CPU
	int N = 128;
	int BLOCKSIZE = BSZ;

	cudaSetDevice(2);

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
	dim3 dimGrid(int((N-2-0.5)/BLOCKSIZE)+1, int((N-2-0.5)/BLOCKSIZE)+1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
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

/* Solve del^2 u = -8*pi^2*sin(2*pi*x)*sin(2*pi*y)
With Dirichlet BCs = 1 on 0<x<1, 0<y<1.
Analytical solution: u = sin(2*pi*x)*sin(2*pi*y) + 1

*/

#include <cusp/gallery/poisson.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <cusp/array1d.h>
#include <cusp/precond/smoothed_aggregation.h>

double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

int main()
{
	int N = 100; // Nodes per side
	float xmin = 0.0f;
	float xmax = 1.0f;
	float ymin = 0.0f;

	float h = (xmax - xmin)/(float)(N-1);
	
	// Generate mesh (if plotting and for RHS)
	cusp::array1d<float, cusp::host_memory> x(N*N, 0);
	cusp::array1d<float, cusp::host_memory> y(N*N, 0);
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	x[j*N+i] = xmin + i*h;
			y[j*N+i] = ymin + j*h;
		}
	}

	// Generate matrix
	cusp::coo_matrix<int, float, cusp::host_memory> A;
	cusp::gallery::poisson5pt(A, N-2, N-2);

	// Generate RHS, solution vector, and analytical solution
	cusp::array1d<float, cusp::host_memory> b(A.num_rows, 1.0f);
	cusp::array1d<float, cusp::host_memory> u(A.num_rows, 0);
	cusp::array1d<float, cusp::host_memory> u_an(A.num_rows, 0);

	for (int j=1; j<N-1; j++)
	{	for (int i=1; i<N-1; i++)
		{	
			b[(j-1)*(N-2)+(i-1)] = 8*M_PI*M_PI*sin(2*M_PI*x[j*N+i])*sin(2*M_PI*y[j*N+i])*h*h;
			u_an[(j-1)*(N-2)+(i-1)] = sin(2*M_PI*x[j*N+i])*sin(2*M_PI*y[j*N+i]) + 1.0f;
			if ((j==1) || (j==N-2))
			{	b[(j-1)*(N-2)+(i-1)] += 1.0f;
			} 
			if ((i==1) || (i==N-2))
			{	b[(j-1)*(N-2)+(i-1)] += 1.0f;
			} 
		}
	}

	float tol = 1e-5;
	int max_iter = 1000;

	// Setup monitor
	
	// Setup preconditioner (identity preconditioner is equivalent to nothing!)

	// Solve 

	// Look at errors
	float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
	for (int j=1; j<N-1; j++)
	{	for (int i=1; i<N-1; i++)
		{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
			L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
		}
	}

	L2_error = sqrt(L2_1/L2_2);

	std::cout<<L2_error<<std::endl;
}

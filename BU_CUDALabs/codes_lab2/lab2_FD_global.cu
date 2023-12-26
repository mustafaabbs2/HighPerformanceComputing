/*** Heat equation with FD in global memory ***/
#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>

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

// double get_time() 
// {  struct timeval tim;
//   cudaThreadSynchronize();
//   gettimeofday(&tim, NULL);
//   return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
// }

// GPU kernels
__global__ void copy_array (float *u, float *u_prev, int N, int BSZ)
{	/******* write your kernel here! ***/
}

__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha, int BSZ)
{	/***** write your kernel here! ***/
}

int main()
{
	// Allocate in CPU
	int N = 128;
	int BLOCKSIZE = 16;

	float xmin 	= 0.0f;
	float xmax 	= 3.5f;
	float ymin 	= 0.0f;
	//float ymax 	= 2.0f;
	float h   	= (xmax-xmin)/(N-1);
	float dt	= 0.00001f;	
	float alpha	= 0.645f;
	float time 	= 0.4f;

	int steps = ceil(time/dt);
	int I;

	float *x  	= new float[N*N]; 
	float *y  	= new float[N*N]; 
	float *u  	= new float[N*N];
	float *u_prev  	= new float[N*N];


	// Generate mesh and intial condition
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			x[I] = xmin + h*i;
			y[I] = ymin + h*j;
			u[I] = 0.0f;
			if ( (i==0) || (j==0)) 
				{u[I] = 200.0f;}
		}
	}

	// Allocate in GPU

	// Copy to GPU

	// Loop 
	dim3 dimGrid(); // number of blocks?
	dim3 dimBlock(); // threads per block?
	for (int t=0; t<steps; t++)
	{	copy_array <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, BLOCKSIZE);
		update <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha, BLOCKSIZE);
	}
	
	// Copy result back to host

	std::ofstream temperature("temperature_global.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
			temperature<<x[I]<<"\t"<<y[I]<<"\t"<<u[I]<<std::endl;
		}
		temperature<<"\n";
	}

	temperature.close();

	// Free device
	cudaFree(u_d);
	cudaFree(u_prev_d);
}

#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>


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

double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

// GPU kernels
__global__ void copy_array(float *u, float *u_prev, int N)
{
        int i = threadIdx.x;
        int j = threadIdx.y;
        int I = blockIdx.y*BSZ*N + blockIdx.x*BSZ + j*N + i;
        if (I>=N*N){return;}    
        u_prev[I] = u[I];

}

__global__ void update (float *u, float *u_prev, int N, float h, float dt, float alpha)
{
	// Setting up indices
	int i = threadIdx.x;
	int j = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int I_0 = by*N*BSZ + bx*BSZ; //Index of the first element of each block including halo
	int Index = by*BSZ*N + bx*BSZ + (j+1)*N + i+1; // Global index without halo

	int G_I = bx*BSZ+i+1; // Global I index
	int G_J = by*BSZ+j+1; // Global J index
	

	__shared__ float u_prev_sh[BSZ+2][BSZ+2];

	// Local indices inside block including halo
	int ii = j*BSZ + i; // Flatten thread indexing
	int I = ii%(BSZ+2); // x-direction index including halo
	int J = ii/(BSZ+2); // y-direction index including halo
	
	int I_n = I_0 + J*N + I; //General index

	// More data than threads-> Load to shared memory in two steps  
	// First step
	u_prev_sh[I][J] = u_prev[I_n];

	// Second step
	int ii2 = BSZ*BSZ + j*BSZ + i;
	int I2  = ii2%(BSZ+2);
	int J2  = ii2/(BSZ+2);

	int I_n2 = I_0 + J2*N + I2; //General index

	if ( (I2<(BSZ+2)) && (J2<(BSZ+2)) && (ii2 < N*N) )
		u_prev_sh[I2][J2] = u_prev[I_n2]; 

	__syncthreads();

	//if (Index>=N*N-1){return;}	
	if ( (G_J>=N-1) || (G_I>=N-1)){return;}	
	{	u[Index] = u_prev_sh[i+1][j+1] + alpha*dt/h/h * (u_prev_sh[i+2][j+1] + u_prev_sh[i][j+1] + u_prev_sh[i+1][j+2] + u_prev_sh[i+1][j] - 4*u_prev_sh[i+1][j+1]);
	}
	
	// Boundary conditions are automatically imposed
	// as we don't touch boundaries
}

int main()
{
	// Allocate in CPU
	int N = 128;
	int BLOCKSIZE = BSZ;

	cudaSetDevice(1);

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
	float *u_d, *u_prev_d;
	
	cudaMalloc( (void**) &u_d, N*N*sizeof(float));
	cudaMalloc( (void**) &u_prev_d, N*N*sizeof(float));

	// Copy to GPU
	cudaMemcpy(u_d, u, N*N*sizeof(float), cudaMemcpyHostToDevice);

	// Loop 
	dim3 dimGrid(int((N-2-0.5)/BLOCKSIZE)+1, int((N-2-0.5)/BLOCKSIZE)+1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	double start = get_time();
	for (int t=0; t<steps; t++)
	{	copy_array <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N);
		update <<<dimGrid, dimBlock>>> (u_d, u_prev_d, N, h, dt, alpha);
	}

	double stop = get_time();
	double time1 = stop - start;
	std::cout<<"time = "<<time1<<std::endl;

	checkErrors("update");
	
	// Copy result back to host
	cudaMemcpy(u, u_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	std::ofstream temperature("temperature_ghost.txt");
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	I = N*j + i;
	//		std::cout<<u[I]<<"\t";
			temperature<<x[I]<<"\t"<<y[I]<<"\t"<<u[I]<<std::endl;
		}
		temperature<<"\n";
	//	std::cout<<std::endl;
	}

	temperature.close();

	// Free device
	cudaFree(u_d);
	cudaFree(u_prev_d);
}

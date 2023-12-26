// A*At = B
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#define BLOCKSIZE 16

typedef struct
{	int width;
	int height;
	float *elements;
} Matrix;

double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

__global__ void transpose(Matrix A, Matrix B, int W, int H, int MAX)
{	/**** write your kernel here! ******/
}


// Host code
int main()
{
	int W = 16; // matrix width
	int H = 16; // matrix height

	int W_max = ((W-0.5)/BLOCKSIZE + 1)*BLOCKSIZE;
	int H_max = ((H-0.5)/BLOCKSIZE + 1)*BLOCKSIZE;

	int MAX;
	if (W_max/H_max == 0){MAX = H_max;}
	else MAX = W_max;

	Matrix A, B;
	
	A.width = W;
	B.width = H;
	A.height = H;
	B.height = W;

	A.elements = new float [MAX*MAX];
	B.elements = new float [MAX*MAX];

	int size = MAX*MAX*sizeof(float);

	// Initialize matrix
	for (int j=0; j<MAX; j++)
	{	for (int i=0; i<MAX; i++)
		{	A.elements[j*MAX+i] = 0.0f;
		}
	}

	// Fill up matrix
        std::ifstream A_input;
        A_input.open("A.txt");

        float a;
        A_input >> a;
        while (!A_input.eof())  
        {       for (int j=0; j<H; j++)
                {       for (int i=0; i<W; i++)
                        {       A.elements[j*MAX+i] = a;
                                A_input >> a;
                        }   
                }   
        }   
        A_input.close();


 	// Allocate in GPU
	Matrix A_d, B_d;

	A_d.width = W;
	B_d.width = H;
	A_d.height = H;
	B_d.height = W;

	cudaMalloc( (void**) &A_d.elements, size);
	cudaMalloc( (void**) &B_d.elements, size);

	cudaMemcpy(A_d.elements, A.elements, size, cudaMemcpyHostToDevice);

	dim3 dimBlock (BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid (MAX/BLOCKSIZE, MAX/BLOCKSIZE);
	double start = get_time();
	transpose<<<dimGrid, dimBlock>>>(A_d, B_d, W, H, MAX);
	double stop = get_time();

	double time = stop-start;

	cudaMemcpy(B.elements, B_d.elements, size, cudaMemcpyDeviceToHost);

	// Print results
	for (int j=0; j<H; j++)
	{	for (int i=0; i<H; i++)
			std::cout<<B.elements[j*MAX + i]<<"\t";
		std::cout<<std::endl;
	}

	std::cout<<time<<std::endl;
	cudaFree(A_d.elements);	
	cudaFree(B_d.elements);	
	
}

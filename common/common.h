#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <fstream> 


//Functions Headers from Udemy

enum INIT_PARAM
{
	INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
};

//simple initialization
void initialize(int * input, const int array_size,
	INIT_PARAM PARAM = INIT_ONE_TO_TEN, int x = 0);

void initialize(float * input, const int array_size,
	INIT_PARAM PARAM = INIT_ONE_TO_TEN);


//compare two arrays
void compare_arrays(int * a, int * b, int size);

//reduction in cpu
int reduction_cpu(int * input, const int size);

//compare results
void compare_results(int gpu_result, int cpu_result);

//print array
void print_array(int * input, const int array_size);

//print array
void print_array(float * input, const int array_size);

//print matrix
void print_matrix(int * matrix, int nx, int ny);

void print_matrix(float * matrix, int nx, int ny);

//get matrix
int* get_matrix(int rows, int columns);

//matrix transpose in CPU
void mat_transpose_cpu(int * mat, int * transpose, int nx, int ny);

//print_time_using_host_clock
void print_time_using_host_clock(clock_t start, clock_t end);

void printData(char *msg, int *in, const int size);

void compare_arrays(float * a, float * b, float size);

void sum_array_cpu(float* a, float* b, float *c, int size);

void print_arrays_toafile(int*, int , char* );

void print_arrays_toafile_side_by_side(float*,float*,int,char*);

void print_arrays_toafile_side_by_side(int*, int*, int, char*);


//Functions from Professional CUDA Programming 


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


inline void print_time(clock_t start, clock_t end)
{
	printf("GPU kernel execution time : %4.6f \n",
		(double)((double)(end - start) / CLOCKS_PER_SEC));
}


inline void print_array(int * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%d,", input[i]);
		}
		else
		{
			printf("%d \n", input[i]);
		}
	}
}

inline void compare_arrays(float * a, float * b, float size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			
			return;
		}
	}
	printf("Arrays are the same \n");
}


#endif // !COMMON_H
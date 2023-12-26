#include<iostream>


__global__ void unique_gid_1d(int* input) {

    int gid =  blockIdx.x * blockDim.x + threadIdx.x;// number threads in a block in a row-major form 

    printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);

}

__global__ void unique_gid_1d_withSize(int* input, int size) {

    int gid =  blockIdx.x * blockDim.x + threadIdx.x;// number threads in a block in a row-major form

    if (gid < size)
    printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);

}



int main (void){

int size= 150;
int byte_size= size*sizeof(int);


int *h_input;
h_input = (int*) malloc(byte_size);


//Don't worry about this - just to init array with random value 
time_t t; 
srand((unsigned) time(&t));

for(int i =0; i < size; i++)
{
    h_input[i] = (int)(rand() & 0xff);
}


//allocating memory on the device
int * d_input; 
cudaMalloc((void**)&d_input, byte_size);

cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);


dim3 block(32); //block with 64 threads in each direction
dim3 grid(5); //two blocks 

unique_gid_1d_withSize<<<grid,block>>>(d_input, size);
cudaDeviceSynchronize();


cudaFree(d_input);
free(h_input);


cudaDeviceReset();

return 0;

}
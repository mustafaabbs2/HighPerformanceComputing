#include<iostream>


__global__ void kernel() {
    printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d" ,
    threadIdx.x, threadIdx.y, threadIdx.z);

}


int main (void){

int nx, ny;
nx = 16;
ny = 16; //total 256 threads

//dim3 block(8,8); //block with 8 threads in each direction
//dim3 grid(nx/block.x, ny/block.y); //two blocks in each direction

kernel<<<1,1>>>();
cudaDeviceSynchronize();
cudaDeviceReset();

return 0;

}
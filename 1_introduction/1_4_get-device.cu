#include<iostream>
int main (void){
    
    cudaDeviceProp prop;
    int count=5;
    cudaGetDeviceCount(&count) ;

    cudaError_t rc, rd; 

    int driver_version = 0, runtime_version = 0;

    rc = cudaDriverGetVersion(&driver_version);
    printf("Driver Version %d \n", driver_version);
    rd = cudaRuntimeGetVersion(&runtime_version);
    printf("Runtime Version %d \n", runtime_version);


    for (int i=0; i< count; i++) {
         cudaGetDeviceProperties( &prop, i );
          printf( " --- General Information for device %d ---\n", i );
          printf( "Name: %s\n", prop.name );
          printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
          printf( "Clock rate: %d\n", prop.clockRate );
          printf( " --- Memory Information for device %d ---\n", i );
          printf( "Total global mem: %ld\n", prop.totalGlobalMem );
          printf( "Total constant Mem: %ld\n", prop.totalConstMem );
          printf( "Max mem pitch: %ld\n", prop.memPitch );
          printf( "Texture Alignment: %ld\n", prop.textureAlignment );
          printf( " --- MP Information for device %d ---\n", i );
          printf( "Multiprocessor count: %d\n",
          prop.multiProcessorCount );
          printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
          printf( "Registers per mp: %d\n", prop.regsPerBlock );
          printf( "Threads in warp: %d\n", prop.warpSize );
          printf( "Max threads per block: %d\n",
          prop.maxThreadsPerBlock );
          printf( "Max thread dimensions: (%d, %d, %d)\n",
          prop.maxThreadsDim[0], prop.maxThreadsDim[1],
          prop.maxThreadsDim[2] );
          printf( "Max grid dimensions: (%d, %d, %d)\n",
          prop.maxGridSize[0], prop.maxGridSize[1],
          prop.maxGridSize[2] );
          printf( "\n" );
    }
    return 0;

}

#pragma once

__global__ void copy_array(float* u, float* u_prev, int N, int BSZ);
__global__ void copy_kernel(float* u, float* u_prev, int N, int BSZ, int N_max);

#pragma once
#include <vector>

void updateCPU(std::vector<float>& u, std::vector<float>& u_prev, size_t N, float h, float dt, float alpha);
// void updateGPU(float* u_d,
// 			   float* u_prev_d,
// 			   int N,
// 			   float h,
// 			   float dt,
// 			   float alpha,
// 			   int BSZ,
// 			   dim3 dimGrid,
// 			   dim3 dimBlock);
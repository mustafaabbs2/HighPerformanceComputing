#pragma once
#include <cuda_runtime.h>
#include <vector>

void update(
	std::vector<float>& u, std::vector<float>& u_prev, size_t N, float h, float dt, float alpha);

void update_(float* u_d,
			 float* u_prev_d,
			 int N,
			 float h,
			 float dt,
			 float alpha,
			 int BSZ,
			 dim3 dimGrid,
			 dim3 dimBlock);
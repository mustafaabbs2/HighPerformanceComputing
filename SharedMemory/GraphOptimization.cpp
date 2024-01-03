#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>
//Solving a graph problem with min-plus matrix multiplication
// c_ij = min{a_ik + b_kj}, with k from 1 -> n
// This is Floyd's algorithm

//unoptimized
void step_v0(float* r, const float* d, int n)
{
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			float v = std::numeric_limits<float>::infinity();
			for(int k = 0; k < n; ++k)
			{
				float x = d[n * i + k];
				float y = d[n * k + j];
				float z = x + y;
				v = std::min(v, z);
			}
			r[n * i + j] = v;
		}
	}
}

//openmp - speedup of factor 3.9
void step_v1(float* r, const float* d, int n)
{
#pragma omp parallel for
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			float v = std::numeric_limits<float>::infinity();
			for(int k = 0; k < n; ++k)
			{
				float x = d[n * i + k];
				float y = d[n * k + j];
				float z = x + y;
				v = std::min(v, z);
			}
			r[n * i + j] = v;
		}
	}
}

//Linear reading + OpenMP

void step_v2(float* r, const float* d, int n)
{
	std::vector<float> t(n * n);

#pragma omp parallel for
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			t[n * j + i] = d[n * i + j];
		}
	}

#pragma omp parallel for
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			float v = std::numeric_limits<float>::infinity();
			for(int k = 0; k < n; ++k)
			{
				float x = d[n * i + k];
				float y = t[n * j + k];
				float z = x + y;
				v = std::min(v, z);
			}
			r[n * i + j] = v;
		}
	}
}

constexpr float infty = std::numeric_limits<float>::infinity();

void step_v3(float* r, const float* d_, int n)
{
	constexpr int nb = 4;
	int na = (n + nb - 1) / nb;
	int nab = na * nb;

	// input data, padded
	std::vector<float> d(n * nab, infty);
	// input data, transposed, padded
	std::vector<float> t(n * nab, infty);

#pragma omp parallel for
	for(int j = 0; j < n; ++j)
	{
		for(int i = 0; i < n; ++i)
		{
			d[nab * j + i] = d_[n * j + i];
			t[nab * j + i] = d_[n * i + j];
		}
	}

#pragma omp parallel for
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			// vv[0] = result for k = 0, 4, 8, ...
			// vv[1] = result for k = 1, 5, 9, ...
			// vv[2] = result for k = 2, 6, 10, ...
			// vv[3] = result for k = 3, 7, 11, ...
			float vv[nb];
			for(int kb = 0; kb < nb; ++kb)
			{
				vv[kb] = infty;
			}
			for(int ka = 0; ka < na; ++ka)
			{
				for(int kb = 0; kb < nb; ++kb)
				{
					float x = d[nab * i + ka * nb + kb];
					float y = t[nab * j + ka * nb + kb];
					float z = x + y;
					vv[kb] = std::min(vv[kb], z);
				}
			}
			// v = result for k = 0, 1, 2, ...
			float v = infty;
			for(int kb = 0; kb < nb; ++kb)
			{
				v = std::min(vv[kb], v);
			}
			r[n * i + j] = v;
		}
	}
}
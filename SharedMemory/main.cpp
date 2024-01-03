#include <chrono>
#include <iostream>

#include "GraphOptimization.h"

int main()
{
	constexpr int n = 3;
	const float d[n * n] = {
		0,
		8,
		2,
		1,
		0,
		9,
		4,
		5,
		0,
	};

	float r[n * n];

	auto start_time = std::chrono::high_resolution_clock::now();
	step_v0(r, d, n);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V0: " << duration.count() << " microseconds" << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	step_v1(r, d, n);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V1: " << duration.count() << " microseconds" << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	step_v2(r, d, n);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V2: " << duration.count() << " microseconds" << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	step_v3(r, d, n);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V2: " << duration.count() << " microseconds" << std::endl;

	//print at the end
	for(int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			std::cout << r[i * n + j] << " ";
		}
		std::cout << "\n";
	}
}
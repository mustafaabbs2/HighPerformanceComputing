#include <chrono>
#include <iostream>

#include "GraphOptimization.h"
#include "PiApproximation.h"

void runGraphTest()
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
	std::cout << "Execution Time V3: " << duration.count() << " microseconds" << std::endl;

#ifdef __GNUC__
	start_time = std::chrono::high_resolution_clock::now();
	step_v4(r, d, n);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V4: " << duration.count() << " microseconds" << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	step_v5(r, d, n);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Execution Time V5: " << duration.count() << " microseconds" << std::endl;

#endif

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

void runPiTest()
{
	size_t numSteps = 100;

	warmup();

	auto start_time = std::chrono::high_resolution_clock::now();
	auto pi = approximatePiSerial(numSteps);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "approximatePiSerial: " << duration.count() << " microseconds" << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	pi = approximatePiParallelNoReduction(numSteps);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "approximatePiParallelNoReduction: " << duration.count() << " microseconds"
			  << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	pi = approximatePiParallel(numSteps);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "approximatePiParallel: " << duration.count() << " microseconds" << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	pi = approximatePiParallelPadded(numSteps);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "approximatePiParallelPadded: " << duration.count() << " microseconds"
			  << std::endl;

	//This is super slow, wtf?
	start_time = std::chrono::high_resolution_clock::now();
	pi = approximatePiStdPar(numSteps);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "approximatePiStdPar: " << duration.count() << " microseconds" << std::endl;

	//EXTREMELY slow
	start_time = std::chrono::high_resolution_clock::now();
	pi = approximatePiParallelThreads(numSteps, 4);
	end_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "approximatePiParallelThreads: " << duration.count() << " microseconds"
			  << std::endl;

	std::cout << "Pi: " << pi << std::endl;
}

int main()
{
	//To test code in GraphOptimizer.cpp
	// runGraphTest();

	//test PI approximation
	runPiTest();
}
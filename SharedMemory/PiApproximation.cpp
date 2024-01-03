#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <vector>

//don't start without warmup if you're timing..
void warmup()
{
	std::vector<int> vectorA(1000, 1);
	std::vector<int> vectorB(1000, 2);
	std::vector<int> vectorC(1000);

#pragma omp parallel
	{
// Perform a simple parallel computation
#pragma omp for
		for(int i = 0; i < 1000; ++i)
		{
			vectorC[i] = vectorA[i] + vectorB[i];
		}
	}
}

double approximatePiSerial(size_t numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);
	double sum = 0.0;

	for(size_t i = 0; i < numSteps; ++i)
	{
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}

	double piApproximation = sum * step;
	return piApproximation;
}

double approximatePiParallel(size_t numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);
	double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
	for(size_t i = 0; i < numSteps; ++i)
	{
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}

	double piApproximation = sum * step;
	return piApproximation;
}

double approximatePiParallelNoReduction(size_t numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);
	double sum = 0.0;

#pragma omp parallel
	{
		double localSum = 0.0;

#pragma omp for
		for(size_t i = 0; i < numSteps; ++i)
		{
			double x = (i + 0.5) * step;
			localSum += 4.0 / (1.0 + x * x);
		}

// #pragma omp critical - a little more heavy
#pragma omp atomic
		sum += localSum;
	}

	double piApproximation = sum * step;
	return piApproximation;
}

struct PaddedDouble
{
	double value;
	char padding
		[64 -
		 sizeof(
			 double)]; // Ensure that each struct is 64 bytes to avoid false sharing - size of a cacheline
};

// Slows it down...
double approximatePiParallelPadded(size_t numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);
	PaddedDouble sum = {0.0};

#pragma omp parallel
	{
		PaddedDouble localSum = {0.0};

#pragma omp for
		for(size_t i = 0; i < numSteps; ++i)
		{
			double x = (i + 0.5) * step;
			localSum.value += 4.0 / (1.0 + x * x);
		}

#pragma omp critical
		sum.value += localSum.value;
	}

	double piApproximation = sum.value * step;
	return piApproximation;
}

//Using standard parallelism
double approximatePiStdPar(size_t numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);

	std::vector<double> localSums(numSteps, 0.0);

	std::for_each(std::execution::par, localSums.begin(), localSums.end(), [&](double& localSum) {
		// Calculate partial sum for each element
		size_t i = &localSum - &localSums[0]; // Calculate the index
		double x = (i + 0.5) * step;
		localSum = 4.0 / (1.0 + x * x);
	});

	double sum = std::accumulate(localSums.begin(), localSums.end(), 0.0);

	double piApproximation = sum * step;
	return piApproximation;
}

double approximatePiParallelThreads(size_t numSteps, int numThreads)
{
	double step = 1.0 / static_cast<double>(numSteps);

	std::vector<std::thread> threads(numThreads);
	std::vector<double> localSums(numThreads, 0.0);

	size_t stepsPerThread = numSteps / numThreads;

	for(int i = 0; i < numThreads; ++i)
	{
		size_t start = i * stepsPerThread;
		size_t end = (i == numThreads - 1) ? numSteps : (i + 1) * stepsPerThread;

		threads[i] = std::thread([&, start, end, i]() {
			for(size_t j = start; j < end; ++j)
			{
				double x = (j + 0.5) * step;
				localSums[i] += 4.0 / (1.0 + x * x);
			}
		});
	}

	// Join threads
	for(int i = 0; i < numThreads; ++i)
	{
		threads[i].join();
	}

	// Accumulate partial sums
	double sum = 0.0;
	for(int i = 0; i < numThreads; ++i)
	{
		sum += localSums[i];
	}

	double piApproximation = sum * step;
	return piApproximation;
}

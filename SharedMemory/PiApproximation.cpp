#include <cmath>
#include <iostream>
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

double approximatePiSerial(long long numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);
	double sum = 0.0;

	for(long long i = 0; i < numSteps; ++i)
	{
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}

	double piApproximation = sum * step;
	return piApproximation;
}

double approximatePiParallel(long long numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);
	double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
	for(long long i = 0; i < numSteps; ++i)
	{
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}

	double piApproximation = sum * step;
	return piApproximation;
}

double approximatePiParallelNoReduction(long long numSteps)
{
	double step = 1.0 / static_cast<double>(numSteps);
	double sum = 0.0;

#pragma omp parallel
	{
		double localSum = 0.0;

#pragma omp for
		for(long long i = 0; i < numSteps; ++i)
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

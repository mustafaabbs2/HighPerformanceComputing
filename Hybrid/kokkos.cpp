#include "kokkos.h"
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <typeinfo>
namespace Core
{

//functors
struct hello_world
{
	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const
	{
		printf("Hello from i = %i\n", i);
	}
};

struct vector_sum
{
	//constructor
	vector_sum(const std::vector<int>& vec)
		: inputVector(vec)
	{ }

	KOKKOS_INLINE_FUNCTION void operator()(const int i, int& lsum) const
	{

		lsum += inputVector[i];
	}

	std::vector<int> inputVector;
};

//end functors

void init()
{

	Kokkos::initialize();
}

void finalize()
{
	Kokkos::finalize();
}

void getDevice()
{
	auto execution_space = Kokkos::DefaultExecutionSpace();
	std::cout << "Execution space: " << execution_space.name() << std::endl;
}

void helloWorld()
{

	printf("Hello World on Kokkos execution space %s\n", Kokkos::DefaultExecutionSpace().name());
	Kokkos::parallel_for("helloWorld", 15, hello_world());
}

void vectorSum()
{
	int result;
	std::vector<int> ones(1000, 1);
	Kokkos::parallel_reduce(
		"vector_sum", 1000, [=](int i, int& lsum) { lsum += ones[i]; }, result);
	std::cout << "Sum (lambda): " << result << std::endl;

	Kokkos::parallel_reduce("vector_sum", 1000, vector_sum(ones), result);
	std::cout << "Sum (functor): " << result << std::endl;
}

//Part 1: In CPU execution space
void daxpy()
{
	int a = 5;
	int N = 1000;
	int* x = new int[N];
	int* y = new int[N];

	for(int i = 0; i < N; i++)
	{
		x[i] = 1;
		y[i] = 20;
	}

	Kokkos::parallel_for("daxpy", 1000, [=](int i) { y[i] = a * x[i] + y[i]; });
	std::cout << "First element: " << y[0] << std::endl;
}

//Part 2: What if this was on the GPU?
// Using views instead of raw pointers
void daxpy_views()
{
	int a = 5;
	int N = 1000;

	// Use Kokkos::View for x and y
	Kokkos::View<int*> x("x", N);
	Kokkos::View<int*> y("y", N);

	Kokkos::parallel_for(
		"init_arrays", N, KOKKOS_LAMBDA(int i) {
			x(i) = 1;
			y(i) = 20;
		});
	Kokkos::parallel_for(
		"daxpy", N, KOKKOS_LAMBDA(int i) { y(i) = a * x(i) + y(i); });

	std::cout << "First element: " << y(0) << std::endl;
}

void yAx()
{

	//initialize
	int M = 10;
	int N = 10;

	auto y = static_cast<double*>(std::malloc(M * sizeof(double)));
	auto x = static_cast<double*>(std::malloc(N * sizeof(double)));

	auto A = static_cast<double*>(std::malloc(M * N * sizeof(double)));

	Kokkos::parallel_for("initY", M, [=](int i) { y[i] = 1; });
	Kokkos::parallel_for("initX", N, [=](int i) { x[i] = 1; });

	Kokkos::parallel_for("initA", N, [=](int j) {
		for(int i = 0; i < M; i++)
		{
			A[j * M + i] = 1;
		}
	});

	double result = 0;

	Kokkos::parallel_reduce(
		"yAx",
		N,
		[=](int j, double& lsum) {
			double temp2 = 0;

			for(int i = 0; i < M; i++)
			{
				temp2 += A[j * M + i] * x[i];
			}

			lsum += y[j] * temp2;
		},
		result);

	std::free(y);
	std::free(x);
	std::free(A);

	std::cout << "Result is: " << result << std::endl;
}

} // namespace Core
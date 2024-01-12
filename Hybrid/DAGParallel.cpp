#include <Kokkos_Core.hpp>
#include <iostream>

struct TaskA
{
	KOKKOS_INLINE_FUNCTION
	void operator()() const
	{
		std::cout << "Task A\n";
	}
};

struct TaskB
{
	KOKKOS_INLINE_FUNCTION
	void operator()() const
	{
		std::cout << "Task B\n";
	}
};

struct TaskC
{
	KOKKOS_INLINE_FUNCTION
	void operator()() const
	{
		std::cout << "Task C\n";
	}
};

void launchDAG()
{
	Kokkos::initialize();

//Nope not working
	// // Create tasks
	// TaskA taskA;
	// TaskB taskB;
	// TaskC taskC;

	// // Execute tasks in parallel
	// Kokkos::parallel_invoke(taskA, taskB);

	// // Wait for taskA and taskB to complete before executing taskC
	// Kokkos::parallel_invoke(Kokkos::TaskSingle(TopoOrderedTag{}, taskC));

	Kokkos::finalize();
}

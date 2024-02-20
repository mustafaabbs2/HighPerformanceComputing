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

	// Create tasks
	TaskA taskA;
	TaskB taskB;
	TaskC taskC;
}

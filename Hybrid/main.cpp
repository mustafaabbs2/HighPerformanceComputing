#include "DAGParallel.h"
#include "kokkos.h"
#include <iostream>

int main()
{
	Core::helloWorld();

	launchDAG();

	return 0;
}
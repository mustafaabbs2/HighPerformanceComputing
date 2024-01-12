#include "DAGParallel.h"
#include "kokkos.h"
#include <iostream>

int main()
{

	Core::init();
	Core::helloWorld();

	launchDAG();

	Core::finalize();
	return 0;
}
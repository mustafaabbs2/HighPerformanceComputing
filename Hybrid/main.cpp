#include "DAGParallel.h"
#include "kokkos.h"
#include <iostream>

int main()
{

	Core::init();
	Core::helloWorld();

	// launchDAG();
	Core::vectorSum();
	Core::daxpy();
	Core::daxpy_views();

	Core::yAx();
	Core::finalize();
	return 0;
}
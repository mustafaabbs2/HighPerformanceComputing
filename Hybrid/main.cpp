#include "DAGParallel.h"
#include "kokkos.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{

	std::string arg;
	if(argc >= 2)
	{
		arg = std::string(argv[1]);
	}
	else
	{
		std::cerr << "Missing argument. Usage: ./BenchmarkHybrid cuda|openmp" << std::endl;
		return 1;
	}

	Core::init(arg);
	Core::helloWorld();

	// launchDAG();
	Core::vectorSum();
	Core::daxpy();
	Core::daxpy_views();

	Core::yAx();
	Core::finalize();
	return 0;
}
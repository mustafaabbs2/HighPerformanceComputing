#include <iostream>
#include <mpi.h>

// mpiexec -n 4 .\BenchmarkMPI.exe

void initMPI()
{
	MPI_Init(nullptr, nullptr);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::cout << "Initializing process " << rank << " out of " << size << " processes."
			  << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
}

void finalizeMPI()
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	std::cout << "Finalizing process " << rank << " out of " << size << " processes." << std::endl;
	MPI_Finalize();
}
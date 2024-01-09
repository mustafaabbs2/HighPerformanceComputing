#include <iostream>

extern void printVersions();
extern void cublasVecAdd();
extern void thrustReduce();

int main()
{
	printVersions();

	cublasVecAdd();

	thrustReduce();

	return 0;
}
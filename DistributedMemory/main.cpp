#include <iostream>

extern void initMPI();
extern void finalizeMPI();

int main()
{
	initMPI();

	finalizeMPI();
}
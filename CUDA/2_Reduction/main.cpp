#include <iostream>

extern void sumArray_();
extern void sumReduction_();
extern void matrixMultiply_();

int main()
{
	//a full reduction
	sumArray_();

	//Reduce via partial sums and optimize
	sumReduction_();

	matrixMultiply_();

	return 0;
}
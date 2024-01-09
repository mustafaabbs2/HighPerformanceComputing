#include <iostream>

extern void add_(int a, int b, int* c);
extern void helloFromGPU_();
extern void getDevice_();
extern void checkIndex_();
extern void checkIndex3D_();
extern void unique1D_();
extern void unique2D_();
extern void printWarpIndex_();

int main()
{

	//Hello World
	helloFromGPU_();

	//Device Props
	getDevice_();

	// Addition
	int c;
	add_(2, 7, &c);
	std::cout << "The sum is: " << c << std::endl;

	//Check IDs 1D case
	checkIndex_();

	//Check IDs 3D case
	checkIndex3D_();

	//Check gid 1D w/ value
	unique1D_();

	//Check gid 2D w/0 value
	unique1D_();

	//Check warp index
	// printWarpIndex_();
}
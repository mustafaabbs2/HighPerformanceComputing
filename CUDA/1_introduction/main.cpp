#include <iostream>

extern void add_(int a, int b, int* c);
extern void helloFromGPU_();
extern void getDevice_();

int main()
{

	helloFromGPU_();
	getDevice_();
	int c;
	add_(2, 7, &c);
	std::cout << "The sum is: " << c;
}
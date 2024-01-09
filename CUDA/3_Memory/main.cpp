#include <iostream>

extern void sumArray_(size_t blockSize);

int main()
{
	//a full reduction
	sumArray_(128); //128 blocks
	return 0;
}
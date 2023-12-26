#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

int main(void)
{
thrust::device_vector<int> data(4);
data[0] = 10;
data[1] = 20;
data[2] = 30;
data[3] = 40;
int sum = thrust::reduce(data.begin(), data.end());
std::cout << "sum is " << sum << std::endl;
return 0;
}
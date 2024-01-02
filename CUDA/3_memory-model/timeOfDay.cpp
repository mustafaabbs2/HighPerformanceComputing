#include<iostream>
// #include<time.h>
#include<string>
#include<stdio.h>
#include <windows.h> //need for Sleep()
//Contains self written helper functions
#include "../common/common.h"


const std::string current_date_and_time() 
{
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%M", &tstruct); //returns a string with date and time
    return buf;
}



const double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


int main()
{    
    // std::cout<<current_date_and_time();
    clock_t gpu_start, gpu_end;
 	gpu_start = clock();
    std::cout<<"helloworld";
    Sleep(500); 
 	gpu_end = clock();
    print_time(gpu_start, gpu_end);
    return 0;
}
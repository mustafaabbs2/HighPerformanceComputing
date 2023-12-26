  
# Add with a single thread on the GPU

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# Define CUDA function
mod = SourceModule("""
__global__ void hello()  {
  printf("Hello World ");
}""")

func = mod.get_function("hello")

# Vector size
N = 100

func(block=(10,10,10), grid=(1,1))
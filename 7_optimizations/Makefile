EXECS=7_1_loop_unrolling
CC=nvcc
CCBIN="C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.20.27508\\bin\\Hostx64\\x64"

ifeq ($(shell uname),Linux)
    CCBIN= /usr/bin/gcc
else ifeq ($(shell uname),Windows)
	CCBIN="C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.20.27508\\bin\\Hostx64\\x64"
endif

all: ${EXECS}

7_1_loop_unrolling: 7_1_loop_unrolling.cu
	$(CC) -ccbin $(CCBIN) -o 7_1_loop_unrolling 7_1_loop_unrolling.cu

clean:
	rm ${EXECS}
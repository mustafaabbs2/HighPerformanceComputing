EXECS=1_1_hello-world 1_2_hello-world 1_3_add-2-numbers 1_4_get-device
CC=nvcc

all: ${EXECS}

1_1_hello-world: 1_1_hello-world.c
	gcc -o 1_1_hello-world 1_1_hello-world.c

1_2_hello-world: 1_2_hello-world.cu
	${CC} -o 1_2_hello-world 1_2_hello-world.cu

1_3_add-2-numbers: 1_3_add-2-numbers.cu
	${CC} -o 1_3_add-2-numbers 1_3_add-2-numbers.cu

1_4_get-device: 1_4_get-device.cu
	${CC} -o 1_4_get-device 1_4_get-device.cu

clean:
	rm ${EXECS}
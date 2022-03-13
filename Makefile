LIBS+=-L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -lcurand

#objects:=cusolver_kernel_wrapper.o cusolver_kernel.o main.o
objects:=cusolver_kernel.o curand_kernel.o main.o

all: main.x
main.x: $(objects)
	nvcc -g -O3 -o $@ $^ $(LDFLAGS) $(LIBS)

cusolver_kernel.o: cusolver_kernel.cu
	nvcc -g -O2 -gencode=arch=compute_70,code=sm_70 -c -o $@ $^
curand_kernel.o: curand_kernel.cu
	nvcc -g -O2 -gencode=arch=compute_70,code=sm_70 -c -o $@ $^
main.o: main.cu
	nvcc -g -O2 -gencode=arch=compute_70,code=sm_70 -c -o $@ $^

%.o : %.c
	icc -c -o $*.o $<
clean:
	rm -f ./cu*_kernel_f *.mod *.o

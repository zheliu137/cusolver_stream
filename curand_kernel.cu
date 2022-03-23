#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>
#include <curand.h>
//#include "cuda_settings.h"

void createRandoms(int size, double *h_randomArray){
    curandGenerator_t generator;
    double *randomArray;
    cudaMalloc((void**)&randomArray, size*size*sizeof(double));
    // cudaMallocHost((void**)&h_randomArray, size*size*sizeof(double));
    curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(generator,(int)time(NULL));
    curandGenerateUniformDouble(generator,randomArray,size*size);
    cudaMemcpy(h_randomArray, randomArray, sizeof(double) * size * size, cudaMemcpyDeviceToHost);
}

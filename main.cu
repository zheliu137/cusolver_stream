#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

int  cusolver_c_stream(int N,cuDoubleComplex *A,int nmat);
int  cusolver_c_stream_QR(int N,cuDoubleComplex *A,int nmat);
void createRandoms(int size, double *randomArray);

int main (int argc, char* argv[]){
    cuDoubleComplex *A;
    int N=10;
    if (argc > 1 ){
      N = strtol(argv[1],nullptr,0);
    }
    int nmat = 1;
    A = (cuDoubleComplex *)malloc(pow(N,2)*sizeof(cuDoubleComplex)*nmat);
    /* 
    double *rand1;
    double *rand2;
    int size=N; 
    rand1 = (double *)malloc(pow(size,2)*sizeof(double));
    rand2 = (double *)malloc(pow(size,2)*sizeof(double));
    printf("Generating %d by %d random matrix... \n",N,N);
    for (int l=0;l<nmat;l++){
    createRandoms(N, rand1);
    createRandoms(N, rand2);
    for (int i=0;i<N;i++){
    for (int j=0;j<N;j++){
      A[IDX2F(i,j,N)+l*N*N] = {rand1[i+j*N]+rand1[j+i*N],rand2[i+j*N]-rand2[j+i*N]};
    }
    } 
    } 
    */
    for (int i=0;i<N;i++){
    for (int j=0;j<N;j++){
      A[IDX2F(i,j,N)] = {double(i+j+1.0), 10.0*(i-j)};
    }
    }

    cusolver_c_stream_QR( N, A, nmat);

    }

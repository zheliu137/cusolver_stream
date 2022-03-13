//Example 1. Application Using C and cuBLAS: 1-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "cuda_settings.h"
#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

// void run_eig_wrapper_(const int N, cuDoubleComplex *x);
//void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda);
int  cusolver_c_stream(int N,cuDoubleComplex *A,int nmat);
void createRandoms(int size, double *randomArray);
//void cusolver(int N); 
//void test(); 

int main (int argc, char* argv[]){
    cuDoubleComplex *A;
    int N=10;
    if (argc > 1 ){
      N = strtol(argv[1],nullptr,0);
    }
    int nmat = 1;
    A = (cuDoubleComplex *)malloc(pow(N,2)*sizeof(cuDoubleComplex)*nmat);
    //A = (double *)malloc(pow(N,2)*sizeof(double));
    int size=N; 
    double *rand1;
    double *rand2;
    /* 
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
      //A[IDX2F(i,j,N)] = {double(i*j+1), double(3*i*j*(i-j)-j+i)};
      A[IDX2F(i,j,N)] = {double(i+j+1.0), 10.0*(i-j)};
      //A[IDX2F(i,j,N)] = float(i*j);
      //A[IDX2F(i,j,N)] = {1.0,1.2};
      //printf("%f\n",A[IDX2F(i,j,N)]);
    }
    }

    printf("Success.\n");

    cusolver_c_stream( N, A, nmat);

    }

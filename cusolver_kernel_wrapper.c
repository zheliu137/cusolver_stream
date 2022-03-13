#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#define DEBUG

int cusolver(int N, double *x);
int cusolver_(int N, cuDoubleComplex *x);
//int cusolver_(int *_N);
/*
void run_eig_wrapper_(int *_N, double *x)
{
#ifdef DEBUG
    printf("==== Begin running cublas dot ====\n");
#endif
    run_eig_double_(_N, x);
#ifdef DEBUG
    printf("==== Finish running cublas dot ====\n");
#endif
}
*/

//void run_eig_wrapper_( int N, cuDoubleComplex *x)
void run_eig_wrapper_( int N, double *x)
{
#ifdef DEBUG
    printf("==== Begin running cublas dot ====\n");
#endif
    cusolver(N,x);
#ifdef DEBUG
    printf("==== Finish running cublas dot ====\n");
#endif
}

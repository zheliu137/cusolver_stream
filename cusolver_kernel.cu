#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <algorithm>

#include <cuda_runtime.h>
#include <cusolverDn.h>

//#define DEBUG
#define SINGLERUN
#define nstream 10
#ifdef DEBUG
#define CUSOLVER_CHECK(err) (HandlecusolverError(err, __FILE__, __LINE__))
#define CUDA_CHECK(err) (HandleError(err, __FILE__, __LINE__))
#else
#define CUSOLVER_CHECK(err) (err)
#define CUDA_CHECK(err) (err)
#endif

static void HandlecusolverError(cusolverStatus_t err, const char *file, int line )
{

    if (err != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: %d in %s at line %d, (error-code %d)\n",
                err, file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

static void HandleError(cudaError_t err, const char *file, int line)
{

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                cudaGetErrorString(err), file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

//extern "C"
//{
int cusolver_c_stream(const int m,  cuDoubleComplex *A_, const int nmat_ ) {

    const int lda = m;
    const int nmat = 30;

    cusolverDnHandle_t cusolverH[nstream];
    cudaStream_t stream[nstream];
    syevjInfo_t syevj_params[nstream];

    printf("solving %d %dx%d matrices by Jacobi method with %d streams.\n",nmat,m,m, nstream);
    cuDoubleComplex *A; // matrix should be stored in pinned memory

    CUDA_CHECK(cudaMallocHost((void **)&A,sizeof(cuDoubleComplex)*lda * m * nmat));
    //A = (cuDoubleComplex *)malloc (m*lda * nmat * sizeof (cuDoubleComplex));

    cuDoubleComplex *V; // eigenvectors
    double *W; // eigenvalue
    cuDoubleComplex *AMV; // A*V
    AMV = (cuDoubleComplex *)malloc (m * nmat * sizeof (*AMV));
    //V = (cuDoubleComplex *)malloc (m*lda * nmat * sizeof (*V));
    //W = (double *)malloc (m * nmat * sizeof (double));
    CUDA_CHECK(cudaMallocHost((void **)&V,sizeof(cuDoubleComplex)*lda * m * nmat));
    CUDA_CHECK(cudaMallocHost((void **)&W,sizeof(double) * m * nmat));

    // copy to pinned memory
    printf("Copy matrix to pinned memory.\n");
    for (int i=0;i<nmat;i++) {
      std::copy(A_,A_+lda*m,A+i*lda*m);
    }

    cuDoubleComplex *d_A;
    double *d_W;
    int *devInfo;
    cuDoubleComplex *d_work[nstream];
    int lwork[nstream];
    int info_gpu[nmat];

    /* configuration of syevj  */
    const double tol = 1.e-10;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // step 0: allocate device memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cuDoubleComplex) * lda * m * nmat));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * m * nstream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)*nstream));


    for (int i=0; i < nstream; i++ ) {
       int ist = i;
    /* step 1: create cusolver handle, bind a stream */
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));


   }
    /* step 3: copy A to device */
    CUDA_CHECK(
        cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * lda * m * nmat, cudaMemcpyHostToDevice ));
    /* step 4: query working space of syevj */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH[0]));
    CUSOLVER_CHECK(
          cusolverDnZheevj_bufferSize(cusolverH[0], jobz, uplo, m, 
          &d_A[0], lda, &d_W[0], &lwork[0], syevj_params[0]));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH[0]));

    for (int i=0; i < nstream; i++ ) {
       int ist = i;
       CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_work[ist]), 
                          sizeof(cuDoubleComplex) * lwork[0],stream[ist]));
    }

    int nloop = 1  ;
    int nnn = 5000;
    for (int l=0; l < nloop; l++ ) {
    printf("loop\n");
    //for (int i=0; i < nmat; i++ ) {
    for (int i=0; i < nnn; i++ ) {
    //printf("matrices %d\n",i);
    int ist = i%nstream;
    printf("stream =  %d\n",ist);
    //int ist = 0;
    // printf("start %dth matrix, stream = %u \n", i, stream[ist]);

    //printf("1.1\n");
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH[ist]));

    //printf("1.2\n");
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH[ist], stream[ist]));
    //printf("1.3\n");
    /* step 2: configuration of syevj */
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params[ist]));

    /* default value of tolerance is machine zero */
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params[ist], tol));

    /* default value of max. sweeps is 100 */
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params[ist], max_sweeps));


    /* step 5: compute eigen-pair   */
    CUSOLVER_CHECK(cusolverDnZheevj(cusolverH[ist], jobz, uplo, m, 
                                    &d_A[ist*lda*m], lda, &d_W[ist*m], 
                                    d_work[ist], lwork[0], &devInfo[ist],
                                    syevj_params[ist]));

    //printf("1.4\n");
    //CUDA_CHECK(cudaMemcpyAsync(&V[lda*m*i], &d_A[i*m*lda], 
    //       sizeof(cuDoubleComplex) * lda * m, cudaMemcpyDeviceToHost, stream[ist]));
    //CUDA_CHECK(cudaMemcpyAsync(&W[m*i], &d_W[ist*m], 
    //       sizeof(double) * m, cudaMemcpyDeviceToHost, stream[ist]));

    //CUDA_CHECK(cudaFreeAsync(d_work[ist],stream[ist]));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH[ist]));
    //printf("done. %d\n",i);
    CUDA_CHECK(cudaDeviceSynchronize());
    }
    }

    CUDA_CHECK(cudaMemcpyAsync(&info_gpu[0], &devInfo[0], 
           sizeof(int), cudaMemcpyDeviceToHost, stream[0]));
    printf("%d",info_gpu[0]);
    CUDA_CHECK(cudaDeviceSynchronize());
    /*
    // step 6: check results
    double residual;
    for (int i=0; i < nmat; i++ ) {
       residual = 0.0;
       for (int j=0;j < m; j++) { 
#ifdef DEBUG
           printf("A * V(%d), W(%d) * V (%d)\n",j,j,j);
#endif
           for (int k=0; k < m; k++) { 
               AMV[k] = {0.0,0.0};
               for (int l=0; l < m; l++) { 
                   AMV[k] = cuCadd(AMV[k],
                            cuCmul(A_[k+l*m+i*m*lda], V[i*m*lda+l+j*m]));
               }
#ifdef DEBUG
               printf("%0.2f + %0.2fj ", AMV[k].x, AMV[k].y);
               printf("%0.2f + %0.2fj ", 
                     W[i*m+j]*V[i*m*lda+k+j*m].x, W[i*m+j]*V[i*m*lda+k+j*m].y);
               printf("\n");
#endif
               residual = residual + abs(AMV[k].x-W[i*m+j]*V[i*m*lda+k+j*m].x)+
                                     abs(AMV[k].y-W[i*m+j]*V[i*m*lda+k+j*m].y);
           }
       }
    }

#ifdef SINGLERUN
    printf("residual = %e \n", residual);
#endif
    std::copy(V,V+lda*m*nmat,A_);
    */
    // step 7 free device memory and reset device

    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(W));
    CUDA_CHECK(cudaFreeHost(V));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(devInfo));

    for (int i=0; i < nstream; i++ ) {
      CUDA_CHECK(cudaStreamDestroy(stream[i]));
      CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params[i]));
    //  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH[i]));
    }

    CUDA_CHECK(cudaDeviceReset());


    return EXIT_SUCCESS;
}

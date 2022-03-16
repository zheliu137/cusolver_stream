#include <cstdio>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <algorithm>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define DEBUG
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

int cusolver_c_stream(const int m,  cuDoubleComplex *A_, const int nmat_ ) {

    // properties of matrix
    const int lda = m;
    int nmat = 500;

    // cusolver setting variebles
    cusolverDnHandle_t cusolverH[nstream];
    cudaStream_t stream[nstream];
    syevjInfo_t syevj_params[nmat];

    // eigen storage and workspace
    cuDoubleComplex *A; // matrix should be stored in pinned memory

    CUDA_CHECK(cudaMallocHost((void **)&A,sizeof(cuDoubleComplex)*lda * m * nmat));

    cuDoubleComplex *V; // eigenvectors
    double *W; // eigenvalue
    CUDA_CHECK(cudaMallocHost((void **)&V,sizeof(cuDoubleComplex)*lda * m * nmat));
    CUDA_CHECK(cudaMallocHost((void **)&W,sizeof(double) * m * nmat));

    // copy to pinned memory
    for (int i=0;i<nmat;i++) {
      std::copy(A_,A_+lda*m,A+i*lda*m);
    }

    // device variables
    cuDoubleComplex *d_A;
    double *d_W;
    int *devInfo;
    cuDoubleComplex *d_work[nstream];
    int lwork[nstream];

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
    /* step 1: create cusolver handle, bind a stream */
      CUSOLVER_CHECK(cusolverDnCreate(&cusolverH[i]));
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));
      CUSOLVER_CHECK(cusolverDnSetStream(cusolverH[i], stream[i]));
   }
    /* step 2: copy A to device */
    CUDA_CHECK(
        cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * lda * m * nmat, cudaMemcpyHostToDevice ));
    /* step 3: query working space of syevj */
    // After test, I find lwork only depends on the size of matrix
    // So we calculate lwork and allocate d_work before loop
    CUSOLVER_CHECK(
          cusolverDnZheevj_bufferSize(cusolverH[0], jobz, uplo, m, 
          &d_A[0], lda, &d_W[0], &lwork[0], syevj_params[0]));

    for (int i=0; i < nstream; i++ ) {
       CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_work[i]), 
                          sizeof(cuDoubleComplex) * lwork[0],stream[i]));
    }

    // CUDA timer
    cudaEvent_t start[nstream], stop[nstream];
    for (int i = 0 ; i < nstream; i++) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&stop[i]));
    }
    for (int i=0; i < nstream; i++ ) {
      CUDA_CHECK(cudaEventRecord(start[i],stream[i]));
    }

    // C timer
    std::clock_t c_start = std::clock();
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Main part
    int nloop = 10  ;

    for (int l=0; l < nloop; l++ ) {
    for (int i=0; i < nmat; i++ ) {
    int ist = i%nstream;

    /* step 2: configuration of syevj */
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params[i]));

    /* default value of tolerance is machine zero */
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params[i], tol));

    /* default value of max. sweeps is 100 */
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params[i], max_sweeps));


    /* step 5: compute eigen-pair   */
    CUSOLVER_CHECK(cusolverDnZheevj(cusolverH[ist], jobz, uplo, m, 
                                    &d_A[i*lda*m], lda, &d_W[ist*m], 
                                    d_work[ist], lwork[0], &devInfo[ist],
                                    syevj_params[i]));

    }
    }
    for (int i = 0; i< nstream; i++){
        CUDA_CHECK(cudaEventRecord(stop[i],stream[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // step 7 free device memory and reset device

    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(W));
    CUDA_CHECK(cudaFreeHost(V));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(devInfo));

    // C timer: CPU time
    std::clock_t c_end = std::clock();

    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used in cusolver: " << time_elapsed_ms << " ms\n";

    // C timer: WALL time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    
    printf("Wall Time measured: %.3f seconds.\n", elapsed);

    // CUDA timer
    for (int i=0; i< nstream; i++) {
    CUDA_CHECK(cudaEventSynchronize(stop[i]));
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start[i], stop[i]));
    float t_sum = 0;
    t_sum += elapsed_time;

    printf("The %d stream CUDA event time: %gs\n",i,t_sum/1000.0);
    }

    for (int i=0; i< nstream; i++) {
    //printf("The %d stream CUDA event start/stop time: %gms, %gms\n",i,start[i],stop[i]);
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(stop[i]));
    }


    // destroy streams
    for (int i=0; i < nstream; i++ ) {
      CUDA_CHECK(cudaStreamDestroy(stream[i]));
      CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params[i]));
      CUSOLVER_CHECK(cusolverDnDestroy(cusolverH[i]));
    }

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

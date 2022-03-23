# Introduction

A demo of cusolver stream on solving eigen problem of a large number of small matrix.

# Conclusion

After analyses with Nsight system, I found that both jacobi and QR method are unable to employ multi-stream. As there are unavoidable pageable memory copies in these two function, there is no way to realize overlap calculation.

![cusolver_timeline_from_nsight](https://github.com/zheliu137/cusolver_stream/blob/main/cusolver_nsight.png)

Besides, it is different in [cublas](https://github.com/zheliu137/Batched_cuBLAS) where overlap calculation can be watched in nsight system clearly.
![cublas_timeline_from_nsight](https://github.com/zheliu137/cusolver_stream/blob/main/cublas_nsight.png)

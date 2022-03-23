# Introduction

A demo of cusolver stream on solving eigen problem of a large number of small matrix.

# Conclusion

After analyses with Nsight system, I found that both jacobi and QR method are unable to employ multi-stream. As there are unavoidable pageable memory copies in these two function, which is different from [cublas](https://github.com/zheliu137/cublas_stream)
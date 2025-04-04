#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    long m = 16384, n = 16384;

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    long lda = m, ldb = m;

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_B_custom = nullptr;

    double one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    /* step 2: copy A to device */
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * m));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_custom),
                          sizeof(double) * ldb * n));

    dim3 grida((m + 15) / 16, (m + 15) / 16);
    dim3 gridb((m + 15) / 16, (n + 15) / 16);
    dim3 block(16, 16);

    setInitialValue<<<grida, block>>>(m, m, d_A, lda, one);
    setInitialValueUpper<<<grida, block>>>(m, m, d_A, lda, zero);

    generateUniformMatrixDouble(d_B, ldb, n);

    cudaEvent_t start, stop;
    float time1 = 0, temp_time = 0;

    CUDA_CHECK(cudaMemcpy(d_B_custom, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B_custom, ldb);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_B_custom, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B_custom, ldb);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
    time1 += temp_time;

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[cublas dtrsm] " << "m: " << m << ", n: " << n << ", "
              << "latency: " << time1 << " ms, " << (long)m * m * n / 2 / time1 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[Free memory] " << free_mem() / 1024 / 1024 / 1024 << " GB"
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_B_custom));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_WARPUP 5
#define NUM_REPEAT 20

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    long m = 16384, n = 16384;

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    long lda = m, ldb = m, ldc = m;

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;

    double one = 1;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * m));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * lda * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * ldc * n));

    generateUniformMatrixDouble(d_A, lda, m);
    generateUniformMatrixDouble(d_B, ldb, n);

    cudaEvent_t start, stop;
    float time1 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cublasDtrmm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B, ldb, d_C, ldc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        cublasDtrmm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B, ldb, d_C, ldc);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time1 += temp_time;
    }
    time1 /= NUM_REPEAT;

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[cublas dtrmm] " << "m: " << m << ", n: " << n << ", "
              << "latency: " << time1 << " ms, " << (long)m * m * n / time1 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[Free memory] " << free_mem() / 1024 / 1024 / 1024 << " GB"
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

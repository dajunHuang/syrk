#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_WARPUP 2
#define NUM_REPEAT 2

void gemm(cublasHandle_t cublasH, int n, int k, double alpha, double *A, int lda,
          double *B, int ldb, double beta, double *C, int ldc, int nb) {
    cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, k, &alpha,
                              A, lda, nb, B, ldb, nb * ldb, &beta, C, ldc,
                              nb + nb * ldc, n / nb);

    for (int i = 1; n / nb / i / 2 >= 1; i *= 2) {
        cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, i * nb, i * nb,
                                  k, &alpha, A + i * nb, lda, 2 * i * nb, B, ldb,
                                  2 * i * nb * ldb, &beta, C + i * nb, ldc,
                                  2 * (i * nb + i * nb * ldc), n / nb / i / 2);
        cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, i * nb, i * nb,
                                  k, &alpha, A, lda, 2 * i * nb, B + i * nb * ldb, ldb,
                                  2 * i * nb * ldb, &beta, C + i * nb * ldc, ldc,
                                  2 * (i * nb + i * nb * ldc), n / nb / i / 2);
    }
    return;
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    int n = 16384, k = 16384, nb = 512;

    double const fp64_abs_tol = 1.0e-4f;

    if (argc >= 4) {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        nb = atoi(argv[3]);
    }

    int lda = n, ldb = k, ldc = n;

    assert(n % nb == 0);

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;
    double *d_C_cublas = nullptr;

    double one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    /* step 2: copy A to device */
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * k));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * ldb * n));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * ldc * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_cublas),
                          sizeof(double) * ldc * n));

    generateUniformMatrixDouble(d_A, lda, k);
    generateUniformMatrixDouble(d_B, ldb, n);

    CUDA_CHECK_LAST_ERROR();

    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, k, &one, d_A,
                             lda, d_B, ldb, &zero, d_C_cublas, ldc));

    CUDA_CHECK_LAST_ERROR();

    gemm(cublasH, n, k, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);

    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 gridc((n + 15) / 16, (n + 15) / 16);
    dim3 blockc(16, 16);

    checkValueLower<<<gridc, blockc>>>(n, n, d_C, ldc, d_C_cublas, ldc,
                                       fp64_abs_tol);

    cudaEvent_t start, stop;
    float time1 = 0, time2 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        gemm(cublasH, n, k, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        gemm(cublasH, n, k, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time1 += temp_time;
    }
    time1 /= NUM_REPEAT;

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, k, &one, d_A, lda, d_B,
                    ldb, &zero, d_C, ldc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, k, &one, d_A, lda, d_B,
                    ldb, &zero, d_C, ldc);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time2 += temp_time;
    }
    time2 /= NUM_REPEAT;

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[custom dgemm] " << "n: " << n << ", k: " << k << ", "
              << "latency: " << time1 << " ms, " << (long)2 * n * n * k / time1 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[cublas dgemm] " << "n: " << n << ", k: " << k << ", "
              << "latency: " << time2 << " ms, " << (long)2 * n * n * k / time2 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[Free memory] " << free_mem() / 1024 / 1024 / 1024 << " GB"
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_cublas));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

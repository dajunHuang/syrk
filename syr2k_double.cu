#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_WARPUP 1
#define NUM_REPEAT 2

// C = alpha * A * B^T + alpha * B * A^T + beta * C
// A is n * k col major, B is n * k col major, C is n * n col major
void syr2k(cublasHandle_t cublasH, int n, int k, double alpha, double *A, int lda,
           double *B, int ldb, double beta, double *C, int ldc, int nb) {
    double one = 1;
    int num_block = n / nb;
    int left = n % nb;
    cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k, &alpha,
                              A, lda, nb, B, ldb, nb, &beta, C, ldc, nb + nb * ldc,
                              num_block);
    cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k, &alpha,
                              B, ldb, nb, A, lda, nb, &one, C, ldc, nb + nb * ldc,
                              num_block);
    if (left > 0) {
        int offset = num_block * nb;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, left, left, k, &alpha,
                    A + offset, lda, B + offset, ldb, &beta,
                    C + offset + offset * ldc, ldc);
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, left, left, k, &alpha,
                    B + offset, ldb, A + offset, lda, &one,
                    C + offset + offset * ldc, ldc);
    }

    for (int i = 1; i * nb < n; i *= 2) {
        num_block = n / (2 * i * nb);
        left = n - (num_block * 2 * i * nb);
        cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, i * nb, i * nb,
                                  k, &alpha, A + i * nb, lda, 2 * i * nb, B, ldb,
                                  2 * i * nb, &beta, C + i * nb, ldc,
                                  2 * (i * nb + i * nb * ldc), num_block);
        cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, i * nb, i * nb,
                                  k, &alpha, B + i * nb, ldb, 2 * i * nb, A, lda,
                                  2 * i * nb, &one, C + i * nb, ldc,
                                  2 * (i * nb + i * nb * ldc), num_block);
        if (left > i * nb) {
            int offset_row = i * nb + num_block * (2 * i * nb);
            int offset_col = num_block * (2 * i * nb);
            cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, left - i * nb, i * nb, k,
                        &alpha, A + offset_row, lda, B + offset_col, ldb, &beta,
                        C + offset_row + offset_col * ldc, ldc);
            cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, left - i * nb, i * nb, k,
                        &alpha, B + offset_row, ldb, A + offset_col, lda, &one,
                        C + offset_row + offset_col * ldc, ldc);
        }
    }
    return;
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    int n = 16384, k = 16384, nb = 512;
    int check = 0;

    // double const fp64_abs_tol = 1.0e-4f;

    if (argc >= 5) {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        nb = atoi(argv[3]);
        check = atoi(argv[4]);
    }

    int lda = n, ldb = n, ldc = n;

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;

    double one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * k));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * lda * k));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * ldc * n));

    generateUniformMatrixDouble(d_A, lda, k);
    generateUniformMatrixDouble(d_B, ldb, k);

    // CUDA_CHECK_LAST_ERROR();

    // CUBLAS_CHECK(cublasDsyr2k(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
    //                           &one, d_A, lda, d_B, ldb, &zero, d_C_cublas, ldc));

    // CUDA_CHECK_LAST_ERROR();

    // syr2k(cublasH, n, k, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);

    // CUDA_CHECK(cudaDeviceSynchronize());

    // dim3 gridc((n + 15) / 16, (n + 15) / 16);
    // dim3 blockc(16, 16);

    // checkValueLower<<<gridc, blockc>>>(n, n, d_C, ldc, d_C_cublas, ldc,
    //                                    fp64_abs_tol);

    cudaEvent_t start, stop;
    float time1 = 0, time2 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        syr2k(cublasH, n, k, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        syr2k(cublasH, n, k, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time1 += temp_time;
    }
    time1 /= NUM_REPEAT;

    CUDA_CHECK(cudaDeviceSynchronize());

    if (check) {
        double *d_C_cublas = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_cublas),
                              sizeof(double) * ldc * n));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        for (int i{0}; i < NUM_WARPUP; ++i) {
            cublasDsyr2k(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one,
                         d_A, lda, d_B, ldb, &zero, d_C_cublas, ldc);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i{0}; i < NUM_REPEAT; ++i) {
            CUDA_CHECK(cudaEventRecord(start));

            cublasDsyr2k(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one,
                         d_A, lda, d_B, ldb, &zero, d_C_cublas, ldc);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
            time2 += temp_time;
        }
        time2 /= NUM_REPEAT;
        copy_lower_to_upper(n, d_C, ldc);
        copy_lower_to_upper(n, d_C_cublas, ldc);
        CUDA_CHECK(cudaDeviceSynchronize());
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &sonedouble, d_C, ldc,
                    &snegonedobule, d_C_cublas, ldc, d_C, ldc);
        double norm_custom = snorm(n, n, d_C, ldc),
               norm_cublas = snorm(n, n, d_C_cublas, ldc);
        printf("norm_custom: %.6e, norm_cublas: %.6e, forward error: %.6e\n",
               norm_custom, norm_cublas, norm_custom / norm_cublas);
        std::cout << "[cublas dsyr2k] " << "m: " << n << ", n: " << k << ", "
                  << "latency: " << time2 << " ms, "
                  << ((long)n * k * n * 2) / time2 / 1e9 << " TFLOPS" << std::endl;
        CUDA_CHECK(cudaFree(d_C_cublas));
    }

    std::cout << "[custom dsyr2k] " << "m: " << n << ", n: " << k << ", "
              << "latency: " << time1 << " ms, "
              << ((long)n * k * n * 2) / time1 / 1e9 << " TFLOPS" << std::endl;
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

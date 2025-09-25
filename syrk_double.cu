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

#define NUM_WARPUP 5
#define NUM_REPEAT 20

// C = alpha * A * A^T + beta * C
// A is n * k col major, C is n * n col major
void syrk(cublasHandle_t cublasH, long n, long k, double alpha, double *A, long lda,
          double beta, double *C, long ldc, long nb) {
    long num_block = n / nb;
    long left = n % nb;
    cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k, &alpha,
                              A, lda, nb, A, lda, nb, &beta, C, ldc, nb + nb * ldc,
                              num_block);
    if (left > 0) {
        long offset = num_block * nb;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, left, left, k, &alpha,
                    A + offset, lda, A + offset, lda, &beta,
                    C + offset + offset * ldc, ldc);
    }

    for (long i = 1; i * nb < n; i *= 2) {
        num_block = n / (2 * i * nb);
        left = n - (num_block * 2 * i * nb);
        cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, i * nb, i * nb,
                                  k, &alpha, A + i * nb, lda, 2 * i * nb, A, lda,
                                  2 * i * nb, &beta, C + i * nb, ldc,
                                  2 * (i * nb + i * nb * ldc), num_block);
        if (left > i * nb) {
            long offset_row = i * nb + num_block * (2 * i * nb);
            long offset_col = num_block * (2 * i * nb);
            cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, left - i * nb, i * nb, k,
                        &alpha, A + offset_row, lda, A + offset_col, lda, &beta,
                        C + offset_row + offset_col * ldc, ldc);
        }
    }
    return;
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    long n = 16384, k = 16384, nb = 512;
    int check = 0;

    if (argc >= 5) {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        nb = atoi(argv[3]);
        check = atoi(argv[4]);
    }

    long lda = n, ldc = n;

    double *d_A = nullptr;
    double *d_C = nullptr;

    double one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    /* step 2: copy A to device */
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * k));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * ldc * n));

    generateUniformMatrixDouble(d_A, lda, k);

    cudaEvent_t start, stop;
    float time1 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        syrk(cublasH, n, k, one, d_A, lda, zero, d_C, ldc, nb);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        syrk(cublasH, n, k, one, d_A, lda, zero, d_C, ldc, nb);

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
        cublasDsyrk(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, d_A,
                    lda, &zero, d_C_cublas, ldc);
        CUDA_CHECK(cudaDeviceSynchronize());
        copy_lower_to_upper(n, d_C, ldc);
        copy_lower_to_upper(n, d_C_cublas, ldc);
        CUDA_CHECK(cudaDeviceSynchronize());
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &sonedouble, d_C, ldc,
                    &snegonedobule, d_C_cublas, ldc, d_C, ldc);
        double norm_custom = nrm2(cublasH, n, n, d_C, ldc),
               norm_cublas = nrm2(cublasH, n, n, d_C_cublas, ldc);
        printf("norm_custom: %.6e, norm_cublas: %.6e, forward error: %.6e\n",
               norm_custom, norm_cublas, norm_custom / norm_cublas);
        CUDA_CHECK(cudaFree(d_C_cublas));
    }

    std::cout << "[custom dsyrk] " << "m: " << n << ", n: " << k << ", "
              << "latency: " << time1 << " ms, " << (long)n * n * k / time1 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[Free memory] " << free_mem() / 1024 / 1024 / 1024 << " GB"
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

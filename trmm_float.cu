#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_WARPUP 1
#define NUM_REPEAT 2

// C = alpha * A * B + beta * C
// A is m * m col major Lower triangular, B is m * n col major, C is m * n col major
void trmm(cublasHandle_t cublasH, long m, long n, float alpha, float *A, long lda,
          float *B, long ldb, float beta, float *C, long ldc, long nb) {
    long num_block = m / nb;
    long left = m % nb;
    float one = 1;
    cublasSgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, nb, n, nb, &alpha,
                              A, lda, nb + nb * lda, B, ldb, nb, &beta, C, ldc, nb,
                              num_block);
    if (left > 0) {
        long offset = num_block * nb;
        cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, left, n, left, &alpha,
                    A + offset + offset * lda, lda, B + offset, ldb, &beta,
                    C + offset, ldc);
    }
    for (long i = 1; i * nb < m; i *= 2) {
        num_block = m / (2 * i * nb);
        left = m - (num_block * 2 * i * nb);
        cublasSgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, i * nb, n,
                                  i * nb, &alpha, A + i * nb, lda,
                                  2 * (i * nb + i * nb * lda), B, ldb, 2 * i * nb,
                                  &one, C + i * nb, ldc, 2 * i * nb, num_block);
        if (left > i * nb) {
            long offset_row = i * nb + num_block * (2 * i * nb);
            long offset_col = num_block * (2 * i * nb);
            cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, left - i * nb, n, i * nb,
                        &alpha, A + offset_row + offset_col * lda, lda,
                        B + offset_col, ldb, &one, C + offset_row, ldc);
        }
    }
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    long m = 16384, n = 16384, nb = 512;
    int check = 0;

    if (argc >= 5) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        nb = atoi(argv[3]);
        check = atoi(argv[4]);
    }

    long lda = m, ldb = m, ldc = m;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    float one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * lda * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * ldc * n));

    dim3 grida((m + 15) / 16, (m + 15) / 16);
    dim3 blocka(16, 16);

    generateUniformMatrixFloat(d_A, lda, m);
    generateUniformMatrixFloat(d_B, ldb, n);

    setInitialValueUpper<float><<<grida, blocka>>>(m, m, d_A, lda, 0);

    cudaEvent_t start, stop;
    float time1 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        trmm(cublasH, m, n, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        trmm(cublasH, m, n, one, d_A, lda, d_B, ldb, zero, d_C, ldc, nb);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time1 += temp_time;
    }
    time1 /= NUM_REPEAT;

    CUDA_CHECK(cudaDeviceSynchronize());

    if (check) {
        float *d_C_cublas = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_cublas),
                              sizeof(float) * ldc * n));
        cublasStrmm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B, ldb, d_C_cublas,
                    ldc);
        CUDA_CHECK(cudaDeviceSynchronize());
        float sonedouble = 1.0, snegonedobule = -1.0;
        cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &sonedouble, d_C, ldc,
                    &snegonedobule, d_C_cublas, ldc, d_C, ldc);
        float norm_custom = nrm2(cublasH, m, n, d_C, ldc),
              norm_cublas = nrm2(cublasH, m, n, d_C_cublas, ldc);
        printf("norm_custom: %.6e, norm_cublas: %.6e, forward error: %.6e\n",
               norm_custom, norm_cublas, norm_custom / norm_cublas);
        CUDA_CHECK(cudaFree(d_C_cublas));
    }

    std::cout << "[custom strmm] " << "m: " << m << ", n: " << n << ", "
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

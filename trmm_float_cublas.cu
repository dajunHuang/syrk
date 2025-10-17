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
    cudaStream_t stream = NULL;

    long m = 16384, n = 16384;

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    long lda = m, ldb = m, ldc = m;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    float one = 1;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * lda * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * ldc * n));

    dim3 grida((m + 15) / 16, (m + 15) / 16);
    dim3 blocka(16, 16);

    generateUniformMatrixFloat(d_A, lda, m);
    generateUniformMatrixFloat(d_B, ldb, n);

    setInitialValueUpper<float><<<grida, blocka>>>(m, m, d_A, lda, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time1 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cublasStrmm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B, ldb, d_C, ldc);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        PUSH_RANGE("trmm_float_cublas", i);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));

        cublasStrmm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B, ldb, d_C, ldc);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        POP_RANGE;
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time1 += temp_time;
    }
    time1 /= NUM_REPEAT;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[cublas strmm] " << "m: " << m << ", n: " << n << ", "
              << "latency: " << time1 << " ms, " << (long)m * m * n / time1 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[Free memory] " << free_mem() / 1024 / 1024 / 1024 << " GB"
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

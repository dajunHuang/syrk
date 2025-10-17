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

#define NUM_WARPUP 5
#define NUM_REPEAT 20

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    long n = 16384, k = 16384;

    if (argc >= 3) {
        n = atoi(argv[1]);
        k = atoi(argv[2]);
    }

    long lda = n, ldb = n, ldc = n;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    float one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * lda * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * lda * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(float) * ldc * n));

    generateUniformMatrixFloat(d_A, lda, k);
    generateUniformMatrixFloat(d_B, ldb, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time1 = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        cublasSsyr2k(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one,
            d_A, lda, d_B, ldb, &zero, d_C, ldc);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        PUSH_RANGE("syr2k_float_cublas", i);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));

        cublasSsyr2k(cublasH, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one,
            d_A, lda, d_B, ldb, &zero, d_C, ldc);

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

    std::cout << "[cublas ssyr2k] " << "m: " << n << ", n: " << k << ", "
              << "latency: " << time1 << " ms, "
              << ((long)n * k * n * 2) / time1 / 1e9 << " TFLOPS" << std::endl;
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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

// A * X = alpha * B
// A is m * m col major Lower triangular, B is m * n col major, overwrited by X
void trsm(cublasHandle_t cublasH, int m, int n, float alpha, float *A, int lda, float *B, int ldb,
          int nb) {
    float sonefloat = 1.0, snegonefloat = -1.0;
    if (m <= nb) {
        CUBLAS_CHECK(cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                 A, lda, B, ldb));
        return;
    }

    trsm(cublasH, m / 2, n, alpha, A, lda, B, ldb, nb);

    int left = m - m / 2;
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, left, n, m / 2,
                             &snegonefloat, A + m / 2, lda, B, ldb, &sonefloat,
                             B + m / 2, ldb));

    trsm(cublasH, left, n, alpha, A + m / 2 + m / 2 * lda, lda, B + m / 2, ldb, nb);
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    int m = 16384, n = 16384, nb = 512;
    int check = 0;

    float const fp32_abs_tol = 1.0e-1f;

    if (argc >= 5) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        nb = atoi(argv[3]);
        check = atoi(argv[4]);
    }

    int lda = m, ldb = m;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_B_custom = nullptr;
    float *d_B_cublas = nullptr;

    float one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * lda * m));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_custom),
                          sizeof(float) * ldb * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_cublas),
                          sizeof(float) * ldb * n));

    dim3 grida((m + 15) / 16, (m + 15) / 16);
    dim3 gridb((m + 15) / 16, (n + 15) / 16);
    dim3 block(16, 16);

    setInitialValue<<<grida, block>>>(m, m, d_A, lda, one);
    setInitialValueUpper<<<grida, block>>>(m, m, d_A, lda, zero);

    generateUniformMatrixFloat(d_B, ldb, n);

    // print_device_matrix(d_A, lda, 32, 32);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_B_custom, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_cublas, d_B, ldb * n, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    trsm(cublasH, m, n, one, d_A, lda, d_B_custom, ldb, nb);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUBLAS_CHECK(cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda,
                             d_B_cublas, ldb));
    CUDA_CHECK(cudaDeviceSynchronize());

    checkValue<<<gridb, block>>>(m, n, d_B_custom, ldb, d_B_cublas, ldb,
                                 fp32_abs_tol);

    cudaEvent_t start, stop;
    float time1 = 0, time2 = 0, temp_time = 0;

    CUDA_CHECK(cudaMemcpy(d_B_custom, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    trsm(cublasH, m, n, one, d_A, lda, d_B_custom, ldb, nb);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_B_custom, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    trsm(cublasH, m, n, one, d_A, lda, d_B_custom, ldb, nb);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
    time1 += temp_time;

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_B_cublas, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B_cublas, ldb);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_B_cublas, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B_cublas, ldb);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
    time2 += temp_time;

    if(check) {
        CUDA_CHECK(cudaDeviceSynchronize());
        float sonedouble = 1.0, snegonedobule = -1.0;
        cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &sonedouble, d_B_custom, ldb,
                    &snegonedobule, d_B_cublas, ldb, d_B_custom, ldb);
        float norm_custom = snorm(m, n, d_B_custom, ldb),
            norm_cublas = snorm(m, n, d_B_cublas, ldb);
        printf("norm_custom: %.6e, norm_cublas: %.6e, forward error: %.6e\n",
            norm_custom, norm_cublas, norm_custom / norm_cublas);
    }

    std::cout << "[custom strsm] " << "m: " << m << ", n: " << n << ", "
              << "latency: " << time1 << " ms, " << (long)m * n * n / time1 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[cublas strsm] " << "m: " << m << ", n: " << n << ", "
              << "latency: " << time2 << " ms, " << (long)m * n * n / time2 / 1e9
              << " TFLOPS" << std::endl;
    std::cout << "[Free memory] " << free_mem() / 1024 / 1024 / 1024 << " GB"
              << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

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
void trsm(cublasHandle_t cublasH, long m, long n, double alpha, double *A, long lda,
          double *B, long ldb, long nb) {
    double sonedouble = 1.0, snegonedobule = -1.0;
    if (m <= nb) {
        CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A,
                                 lda, B, ldb));
        return;
    }

    trsm(cublasH, m / 2, n, alpha, A, lda, B, ldb, nb);

    long left = m - m / 2;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, left, n, m / 2,
                             &snegonedobule, A + m / 2, lda, B, ldb, &sonedouble,
                             B + m / 2, ldb));

    trsm(cublasH, left, n, alpha, A + m / 2 + m / 2 * lda, lda, B + m / 2, ldb, nb);
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;

    long m = 16384, n = 16384, nb = 512;
    int check = 0;

    // double const fp64_abs_tol = 1.0e-4f;

    if (argc >= 5) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        nb = atoi(argv[3]);
        check = atoi(argv[4]);
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

    if (check) {
        double *d_B_cublas = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B_cublas),
                              sizeof(double) * ldb * n));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(d_B_cublas, d_B, ldb * n, cudaMemcpyDeviceToDevice));
        cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B_cublas, ldb);
        CUDA_CHECK(cudaDeviceSynchronize());
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &sonedouble, d_B_custom,
                    ldb, &snegonedobule, d_B_cublas, ldb, d_B_custom, ldb);
        double norm_custom = nrm2(cublasH, m, n, d_B_custom, ldb),
               norm_cublas = nrm2(cublasH, m, n, d_B_cublas, ldb);
        printf("norm_custom: %.6e, norm_cublas: %.6e, forward error: %.6e\n",
               norm_custom, norm_cublas, norm_custom / norm_cublas);
        CUDA_CHECK(cudaFree(d_B_cublas));
    }

    std::cout << "[custom dtrsm] " << "m: " << m << ", n: " << n << ", "
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

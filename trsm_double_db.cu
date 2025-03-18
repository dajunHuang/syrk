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
void trsm(cublasHandle_t cublasH1, cublasHandle_t cublasH2, long m, long n, double alpha, double *A, long lda,
          double *B, long ldb, long b, long nb) {
    double sonedouble = 1.0, snegonedobule = -1.0;
    long num_nb = (m + nb - 1) / nb;
    for (long i = 0; i < num_nb; i++) {
        long this_nb = min(nb, m - i * nb);
        long num_b_this_nb = (this_nb + b - 1) / b;
        // printf("nb%d, this_nb: %d, row: %d, col: %d, num_b_this_nb: %d\n", i,
        //        this_nb, i * nb, i * nb, num_b_this_nb);
        for (int j = 0; j < num_b_this_nb; j++) {
            long this_b = min(b, this_nb - j * b);
            // printf("b%d, this_b: %d, row: %d, col: %d\n", j, this_b, i * nb + j * b,
            //        i * nb + j * b);
            CUBLAS_CHECK(cublasDtrsm(cublasH1, CUBLAS_SIDE_LEFT,
                                     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, this_b, n, &sonedouble,
                                     A + i * nb + j * b + (i * nb + j * b) * lda,
                                     lda, B + i * nb + j * b, ldb));
            // printf("trsm: m: %d, n: %d, A[%d, %d], B[%d, %d]\n", this_b, n,
            //        i * nb + j * b, i * nb + j * b, i * nb + j * b, 0);
            CUBLAS_CHECK(
                cublasDgemm(cublasH1, CUBLAS_OP_N, CUBLAS_OP_N,
                            this_nb - (j * b + this_b), n, this_b, &snegonedobule,
                            A + i * nb + j * b + this_b + (i * nb + j * b) * lda,
                            lda, B + i * nb + j * b, ldb, &sonedouble,
                            B + i * nb + j * b + this_b, ldb));
            // printf("gemm0: m: %d, n: %d, k: %d, A[%d, %d], X[%d, %d], B[%d, %d]\n",
            //        this_nb - (j * b + this_b), n, this_b, i * nb + j * b + this_b,
            //        i * nb + j * b, i * nb + j * b, 0, i * nb + j * b + this_b, 0);
        }
        CUBLAS_CHECK(cublasDgemm(
            cublasH1, CUBLAS_OP_N, CUBLAS_OP_N, m - (i * nb + this_nb), n, this_nb,
            &snegonedobule, A + i * nb + this_nb + (i * nb) * lda, lda, B + i * nb,
            ldb, &sonedouble, B + i * nb + this_nb, ldb));
        // printf("gemm1: m: %d, n: %d, k: %d, A[%d, %d], X[%d, %d], B[%d, %d]\n",
        //        m - (i * nb + this_nb), n, this_nb, i * nb + this_nb, i * nb, i * nb,
        //        0, i * nb + this_nb, 0);
    }
}

int main(int argc, char *argv[]) {
    
    long m = 32, n = 32, b = 8, nb = 16;
    int check = 0;
    
    if (argc >= 6) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        b = atoi(argv[3]);
        nb = atoi(argv[4]);
        check = atoi(argv[5]);
    }
    
    assert(nb % b == 0);
    
    long lda = m, ldb = m;
    
    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_B_custom = nullptr;
    
    double one = 1, zero = 0;
    
    cublasHandle_t cublasH1, cublasH2;
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&cublasH1));
    CUBLAS_CHECK(cublasCreate(&cublasH2));
    CUBLAS_CHECK(cublasSetStream(cublasH1, stream1));
    CUBLAS_CHECK(cublasSetStream(cublasH2, stream2));

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

    // trsm(cublasH1, cublasH2, m, n, one, d_A, lda, d_B_custom, ldb, b, nb);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(d_B_custom, d_B, ldb * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    trsm(cublasH1, cublasH2, m, n, one, d_A, lda, d_B_custom, ldb, b, nb);

    CUDA_CHECK(cudaDeviceSynchronize());
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
        cublasDtrsm(cublasH1, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, m, n, &one, d_A, lda, d_B_cublas, ldb);
        CUDA_CHECK(cudaDeviceSynchronize());
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDgeam(cublasH1, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &sonedouble, d_B_custom,
                    ldb, &snegonedobule, d_B_cublas, ldb, d_B_custom, ldb);
        double norm_custom = snorm(m, n, d_B_custom, ldb),
               norm_cublas = snorm(m, n, d_B_cublas, ldb);
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

    CUBLAS_CHECK(cublasDestroy(cublasH1));
    CUBLAS_CHECK(cublasDestroy(cublasH2));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

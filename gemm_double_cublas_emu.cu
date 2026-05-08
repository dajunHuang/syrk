#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.h"

#define NUM_WARPUP 2
#define NUM_REPEAT 10

int main(int argc, char* argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int m = 32768, n = 32768, k = 32768;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    double one = 1, zero = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUBLAS_CHECK(cublasSetEmulationStrategy(
        cublasH, CUBLAS_EMULATION_STRATEGY_EAGER));  // CUBLAS_EMULATION_STRATEGY_EAGER
                                                     // CUBLAS_EMULATION_STRATEGY_DEFAULT
                                                     // CUBLAS_EMULATION_STRATEGY_PERFORMANT
    CUBLAS_CHECK(cublasSetEmulationSpecialValuesSupport(
        cublasH,
        CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT));  // CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT
                                                          // CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE
                                                          // CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY
                                                          // CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN
    // CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC
    // CUDA_EMULATION_MANTISSA_CONTROL_FIXED
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    cudaEmulationMantissaControl mantissaControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(cublasH, mantissaControl));
    // Primarily for CUDA_EMULATION_MANTISSA_CONTROL_FIXED, default 55.
    // When using CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC, If the number of required mantissa bits
    // exceeds a library defined default (see Default Library Configurations 79 or a user provided
    // maximum number of bits (see cublasSetFixedPointEmulationMaxMantissaBitCount()), the framework
    // dynamically dispatches to native FP64 (too hard to emulate).
    int count = 55;
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(cublasH, count));
    // Only for CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC, Reduce the final precision required
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaBitOffset(cublasH, 0));
    size_t workspace_size =
        getFixedPointWorkspaceSizeInBytes(m, n, k, 1, false, mantissaControl, count);
    void* workspace;
    // CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    // cublasSetWorkspace(cublasH, workspace, workspace_size);
    std::cout << "Workspace size in MB: " << workspace_size / (1024 * 1024) << std::endl;

    int maxMantissaBitCount = -1, mantissaBitOffset = -1;
    CUBLAS_CHECK(cublasGetFixedPointEmulationMaxMantissaBitCount(cublasH, &maxMantissaBitCount));
    CUBLAS_CHECK(cublasGetFixedPointEmulationMantissaBitOffset(cublasH, &mantissaBitOffset));
    std::cout << "MaxMantissaBitCount: " << maxMantissaBitCount
              << " MantissaBitOffset: " << mantissaBitOffset << std::endl;

    int lda = m, ldb = k, ldc = m;

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * lda * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(double) * ldb * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(double) * ldc * n));

    generateUniformMatrixDouble(d_A, lda, k);
    generateUniformMatrixDouble(d_B, ldb, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i{0}; i < NUM_WARPUP; ++i) {
        CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, d_A, lda, d_B,
                                 ldb, &zero, d_C,
                                 ldc));  // CUBLAS_GEMM_ALGO0_TENSOR_OP
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i{0}; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, d_A, lda, d_B,
                                 ldb, &zero, d_C,
                                 ldc));  // CUBLAS_GEMM_ALGO0_TENSOR_OP

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[cublas dgemm] " << "m: " << m << ", n: " << n << ", k: " << k << ", "
              << "latency: " << time << " ms, "
              << "Effective TFLOPS: " << 2.0 * m * n * k / time / 1e9 << " TFLOPS, " << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(workspace));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

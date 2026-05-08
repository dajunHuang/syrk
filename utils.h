#pragma once

#include <cuComplex.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cusolverDn.h>
#include <library_types.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>

#include <array>  // For std::array
#include <cmath>
#include <cstdint>  // For uint32_t
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

// C++ style array container
const std::array<uint32_t, 7> colors = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                                        0x0000ffff, 0x00ff0000, 0x00ffffff};

#define PUSH_RANGE(name, cid)                              \
    do {                                                   \
        nvtxEventAttributes_t eventAttrib = {};            \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[(cid) % colors.size()]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    } while (0)
#define POP_RANGE nvtxRangePop()

// CUDA API error checking
#define CUDA_CHECK(err)                                                   \
    do {                                                                  \
        cudaError_t err_ = (err);                                         \
        if (err_ != cudaSuccess) {                                        \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                       \
        }                                                                 \
    } while (0)

// CUDA API error checking
#define CUDA_CHECK_LAST_ERROR()                                           \
    do {                                                                  \
        cudaError_t err_ = (cudaGetLastError());                          \
        if (err_ != cudaSuccess) {                                        \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                       \
        }                                                                 \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                   \
    do {                                                                      \
        cusolverStatus_t err_ = (err);                                        \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                       \
        }                                                                     \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                   \
    do {                                                                    \
        cublasStatus_t err_ = (err);                                        \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                       \
        }                                                                   \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                   \
    do {                                                                      \
        cusparseStatus_t err_ = (err);                                        \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusparse error");                       \
        }                                                                     \
    } while (0)

// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)
#define CEILDIV(A, B) (((A) + (B) - 1) / (B))

size_t getFixedPointWorkspaceSizeInBytes(int m, int n, int k, int batchCount, bool isComplex,
                                         cudaEmulationMantissaControl mantissaControl,
                                         int maxMantissaBitCount) {
    constexpr double MULTIPLIER = 1.25;

    int mult = isComplex ? 2 : 1;
    int numSlices = CEILDIV(maxMantissaBitCount + 1, 8);
    int max_splitk = CEILDIV(k, 8192);

    int padded_m = ALIGN_TO(m, 1024);
    int padded_n = ALIGN_TO(n, 1024);
    int padded_k = ALIGN_TO(k, 128);
    int num_blocks_k = CEILDIV(k, 64);

    size_t gemm_workspace = sizeof(int8_t) *
                            ((size_t)padded_m * padded_k + (size_t)padded_n * padded_k) * mult *
                            numSlices;
    gemm_workspace += sizeof(int32_t) * ((size_t)padded_m + padded_n) * mult;

    size_t acc_workspace_ver1 = 0;
    if (isComplex) {
        acc_workspace_ver1 += sizeof(double) * (size_t)m * n * mult * mult;
    }
    size_t acc_workspace_ver2 =
        std::min(sizeof(int32_t) * ((size_t)padded_m * padded_n) * mult * mult * numSlices,
                 (size_t)(1LL << 32)) *
        max_splitk;
    gemm_workspace += std::max(acc_workspace_ver1, acc_workspace_ver2);

    size_t adp_workspace = 0;
    if (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC) {
        adp_workspace = sizeof(int32_t) *
                        ((size_t)m * num_blocks_k + (size_t)n * num_blocks_k + (size_t)m * n) *
                        mult;
    }

    constexpr size_t CONSTANT_SIZE = 128 * 1024 * 1024;
    return (size_t)(std::max(gemm_workspace, adp_workspace) * batchCount * MULTIPLIER) +
           CONSTANT_SIZE;
}

// device memory pitch alignment
static const size_t device_alignment = 32;

template <typename T>
__global__ void setInitialValue(long m, long n, T* a, long lda, T val) {
    long i = threadIdx.x + blockDim.x * blockIdx.x;
    long j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        a[i + j * lda] = val;
    }
}

template <typename T>
__global__ void setInitialValueUpper(long m, long n, T* a, long lda, T val) {
    long i = threadIdx.x + blockDim.x * blockIdx.x;
    long j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < j && i < m && j < n) {
        a[i + j * lda] = val;
    }
}

template <typename T>
__global__ void setInitialValueLower(long m, long n, T* a, long lda, T val) {
    long i = threadIdx.x + blockDim.x * blockIdx.x;
    long j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i > j && i < m && j < n) {
        a[i + j * lda] = val;
    }
}

template <typename T>
__global__ void checkValue(long m, long n, T* A, long lda, T* B, long ldb, double tol) {
    long i = threadIdx.x + blockDim.x * blockIdx.x;
    long j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        double const A_val{static_cast<double>(A[i + j * lda])};
        double const B_val{static_cast<double>(B[i + j * ldb])};
        double const diff{A_val - B_val};
        double const diff_val{std::abs(diff)};
        if (isnan(diff_val) || diff_val > static_cast<double>(tol)) {
            printf("A[%ld, %ld] = %f, B[%ld, %ld] = %f\n", i, j, A_val, i, j, B_val);
        }
    }
}

template <typename T>
__global__ void checkValueLower(long m, long n, T* A, long lda, T* B, long ldb, double tol) {
    long i = threadIdx.x + blockDim.x * blockIdx.x;
    long j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i > j && i < m && j < n) {
        double const A_val{static_cast<double>(A[i + j * lda])};
        double const B_val{static_cast<double>(B[i + j * ldb])};
        double const diff{A_val - B_val};
        double const diff_val{std::abs(diff)};
        if (isnan(diff_val) || diff_val > static_cast<double>(tol)) {
            printf("A[%ld, %ld] = %f, B[%ld, %ld] = %f\n", i, j, A_val, i, j, B_val);
        }
    }
}

void generateUniformMatrixDouble(double* dA, long m, long n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen, dA, long(m * n));
}

void generateUniformMatrixFloat(float* dA, long m, long n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, dA, long(m * n));
}

size_t free_mem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

template <typename T>
__global__ void frobenius_norm_kernelDouble(int64_t m, int64_t n, T* A, int64_t lda, T* norm) {
    int64_t idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    for (int64_t i = idx_x; i < m; i += blockDim.x * gridDim.x) {
        for (int64_t j = idx_y; j < n; j += blockDim.y * gridDim.y) {
            T value = A[i + j * lda];
            T value_squared = T(value) * (value);
            atomicAdd(norm, value_squared);
        }
    }
}

template <typename T>
T nrm2(cublasHandle_t cublasH, long m, long n, T* d_A, long lda);
template <>
float nrm2(cublasHandle_t cublasH, long m, long n, float* d_A, long lda) {
    float norm = 0;

    if (lda != m) {
        printf("lda must be equal to m");
    }

    CUBLAS_CHECK(cublasSnrm2(cublasH, m * n, d_A, 1, &norm));

    return norm;
}
template <>
double nrm2(cublasHandle_t cublasH, long m, long n, double* d_A, long lda) {
    double norm = 0;

    if (lda != m) {
        printf("lda must be equal to m");
    }

    CUBLAS_CHECK(cublasDnrm2(cublasH, m * n, d_A, 1, &norm));

    return norm;
}

template <typename T>
__global__ void copy_lower_to_upper_kernel(long n, T* A, long lda) {
    long i = threadIdx.x + blockDim.x * blockIdx.x;
    long j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < n && j < n) {
        if (i < j) A[i + j * lda] = A[j + i * lda];
    }
}

template <typename T>
void copy_lower_to_upper(long n, T* A, long lda) {
    const long BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    copy_lower_to_upper_kernel<<<gridDim, blockDim>>>(n, A, lda);
}

template <typename T>
void print_device_matrix(T* dA, long ldA, long rows, long cols) {
    T matrix;

    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < cols; j++) {
            cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);
            printf("%5.2f ", matrix);
        }
        printf("\n");
    }
}

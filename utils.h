#pragma once

#include <cuComplex.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cusolverDn.h>
#include <library_types.h>
#include <math.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

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

// device memory pitch alignment
static const size_t device_alignment = 32;

template <typename T>
__global__ void setInitialValue(int m, int n, T *a, int lda, T val) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        a[i + j * lda] = val;
    }
}

template <typename T>
__global__ void setInitialValueUpper(int m, int n, T *a, int lda, T val) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < j && i < m && j < n) {
        a[i + j * lda] = val;
    }
}

template <typename T>
__global__ void setInitialValueLower(int m, int n, T *a, int lda, T val) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i > j && i < m && j < n) {
        a[i + j * lda] = val;
    }
}

template <typename T>
__global__ void checkValue(int m, int n, T *A, int lda, T *B, int ldb, double tol) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        double const A_val{static_cast<double>(A[i + j * lda])};
        double const B_val{static_cast<double>(B[i + j * ldb])};
        double const diff{A_val - B_val};
        double const diff_val{std::abs(diff)};
        if (isnan(diff_val) || diff_val > static_cast<double>(tol)) {
            printf("A[%d, %d] = %f, B[%d, %d] = %f\n", i, j, A_val, i, j, B_val);
        }
    }
}

template <typename T>
__global__ void checkValueLower(int m, int n, T *A, int lda, T *B, int ldb,
                                double tol) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i > j && i < m && j < n) {
        double const A_val{static_cast<double>(A[i + j * lda])};
        double const B_val{static_cast<double>(B[i + j * ldb])};
        double const diff{A_val - B_val};
        double const diff_val{std::abs(diff)};
        if (isnan(diff_val) || diff_val > static_cast<double>(tol)) {
            printf("A[%d, %d] = %f, B[%d, %d] = %f\n", i, j, A_val, i, j, B_val);
        }
    }
}

void generateUniformMatrixDouble(double *dA, int m, int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen, dA, long(m * n));
}

void generateUniformMatrixFloat(float *dA, int m, int n) {
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
__global__ void frobenius_norm_kernelDouble(int64_t m, int64_t n, T *A,
                                            int64_t lda, T *norm) {
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
T snorm(long int m, long int n, T *d_A, long int lda) {
    const long int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    T *d_norm;
    cudaMalloc(&d_norm, sizeof(T));
    cudaMemset(d_norm, 0, sizeof(T));

    frobenius_norm_kernelDouble<<<gridDim, blockDim>>>(m, n, d_A, lda, d_norm);

    T norm;
    cudaMemcpy(&norm, d_norm, sizeof(T), cudaMemcpyDeviceToHost);
    norm = sqrtf(norm);

    cudaFree(d_norm);

    return norm;
}

template <typename T>
__global__ void copy_lower_to_upper_kernel(int n, T *A, long int lda) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < n && j < n) {
        if (i < j) A[i + j * lda] = A[j + i * lda];
    }
}

template <typename T>
void copy_lower_to_upper(int n, T *A, long int lda) {
    const long int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    copy_lower_to_upper_kernel<<<gridDim, blockDim>>>(n, A, lda);
}

template <typename T>
void print_device_matrix(T *dA, int ldA, int rows, int cols) {
    T matrix;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);
            printf("%5.2f ", matrix);
        }
        printf("\n");
    }
}

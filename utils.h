/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

// type traits
template <typename T>
struct traits;

template <>
struct traits<float> {
    // scalar type
    typedef float T;
    typedef T S;

    static constexpr T zero = 0.f;
    static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_32F;
#endif

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG>
    inline static T rand(RNG &gen) {
        return (S)gen();
    }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, S f) { return v * f; }
};

template <>
struct traits<double> {
    // scalar type
    typedef double T;
    typedef T S;

    static constexpr T zero = 0.;
    static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_64F;
#endif

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG>
    inline static T rand(RNG &gen) {
        return (S)gen();
    }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, S f) { return v * f; }
};

template <>
struct traits<cuFloatComplex> {
    // scalar type
    typedef float S;
    typedef cuFloatComplex T;

    static constexpr T zero = {0.f, 0.f};
    static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_32F;
#endif

    inline static S abs(T val) { return cuCabsf(val); }

    template <typename RNG>
    inline static T rand(RNG &gen) {
        return make_cuFloatComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCaddf(a, b); }
    inline static T add(T a, S b) { return cuCaddf(a, make_cuFloatComplex(b, 0.f)); }

    inline static T mul(T v, S f) { return make_cuFloatComplex(v.x * f, v.y * f); }
};

template <>
struct traits<cuDoubleComplex> {
    // scalar type
    typedef double S;
    typedef cuDoubleComplex T;

    static constexpr T zero = {0., 0.};
    static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_64F;
#endif

    inline static S abs(T val) { return cuCabs(val); }

    template <typename RNG>
    inline static T rand(RNG &gen) {
        return make_cuDoubleComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCadd(a, b); }
    inline static T add(T a, S b) { return cuCadd(a, make_cuDoubleComplex(b, 0.)); }

    inline static T mul(T v, S f) { return make_cuDoubleComplex(v.x * f, v.y * f); }
};

template <typename T>
void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <>
void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A,
                  const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <typename T>
void generate_random_matrix(cusolver_int_t m, cusolver_int_t n, T **A, int *lda) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<typename traits<T>::S> dis(-1.0, 1.0);
    auto rand_gen = std::bind(dis, gen);

    *lda = n;

    size_t matrix_mem_size = static_cast<size_t>(*lda * m * sizeof(T));
    // suppress gcc 7 size warning
    if (matrix_mem_size <= PTRDIFF_MAX)
        *A = (T *)malloc(matrix_mem_size);
    else
        throw std::runtime_error("Memory allocation size is too large");

    if (*A == NULL) throw std::runtime_error("Unable to allocate host matrix");

    // random matrix and accumulate row sums
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T *A_row = (*A) + *lda * i;
            A_row[j] = traits<T>::rand(rand_gen);
        }
    }
}

// Makes matrix A of size mxn and leading dimension lda diagonal dominant
template <typename T>
void make_diag_dominant_matrix(cusolver_int_t m, cusolver_int_t n, T *A, int lda) {
    for (int i = 0; i < std::min(m, n); ++i) {
        T *A_row = A + lda * i;
        auto row_sum = traits<typename traits<T>::S>::zero;
        for (int j = 0; j < n; ++j) {
            row_sum += traits<T>::abs(A_row[j]);
        }
        A_row[i] = traits<T>::add(A_row[i], row_sum);
    }
}

// Returns cudaDataType value as defined in library_types.h for the string
// containing type name
cudaDataType get_cuda_library_type(std::string type_string) {
    if (type_string.compare("CUDA_R_16F") == 0)
        return CUDA_R_16F;
    else if (type_string.compare("CUDA_C_16F") == 0)
        return CUDA_C_16F;
    else if (type_string.compare("CUDA_R_32F") == 0)
        return CUDA_R_32F;
    else if (type_string.compare("CUDA_C_32F") == 0)
        return CUDA_C_32F;
    else if (type_string.compare("CUDA_R_64F") == 0)
        return CUDA_R_64F;
    else if (type_string.compare("CUDA_C_64F") == 0)
        return CUDA_C_64F;
    else if (type_string.compare("CUDA_R_8I") == 0)
        return CUDA_R_8I;
    else if (type_string.compare("CUDA_C_8I") == 0)
        return CUDA_C_8I;
    else if (type_string.compare("CUDA_R_8U") == 0)
        return CUDA_R_8U;
    else if (type_string.compare("CUDA_C_8U") == 0)
        return CUDA_C_8U;
    else if (type_string.compare("CUDA_R_32I") == 0)
        return CUDA_R_32I;
    else if (type_string.compare("CUDA_C_32I") == 0)
        return CUDA_C_32I;
    else if (type_string.compare("CUDA_R_32U") == 0)
        return CUDA_R_32U;
    else if (type_string.compare("CUDA_C_32U") == 0)
        return CUDA_C_32U;
    else
        throw std::runtime_error("Unknown CUDA datatype");
}

// Returns cusolverIRSRefinement_t value as defined in cusolver_common.h for the
// string containing solver name
cusolverIRSRefinement_t get_cusolver_refinement_solver(std::string solver_string) {
    if (solver_string.compare("CUSOLVER_IRS_REFINE_NONE") == 0)
        return CUSOLVER_IRS_REFINE_NONE;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_CLASSICAL") == 0)
        return CUSOLVER_IRS_REFINE_CLASSICAL;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_GMRES;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_CLASSICAL_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_CLASSICAL_GMRES;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_GMRES_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_GMRES_GMRES;
    else
        printf("Unknown solver parameter: \"%s\"\n", solver_string.c_str());

    return CUSOLVER_IRS_REFINE_NOT_SET;
}

template <typename T>
bool all_close(std::vector<T> &C, std::vector<T> &C_ref, size_t m, size_t n,
               size_t ldc, T abs_tol, double rel_tol) {
    bool status{true};
    for (size_t j{0U}; j < n; ++j) {
        for (size_t i{j}; i < m; ++i) {
            double const C_val{static_cast<double>(C[i + j * ldc])};
            double const C_ref_val{static_cast<double>(C_ref[i + j * ldc])};
            double const diff{C_val - C_ref_val};
            double const diff_val{std::abs(diff)};
            if (isnan(diff_val) ||
                diff_val >
                    std::max(static_cast<double>(abs_tol),
                             static_cast<double>(std::abs(C_ref_val)) * rel_tol)) {
                std::cout << "C[" << i << ", " << j << "] = " << C_val << " C_ref["
                          << i << ", " << j << "] = " << C_ref_val
                          << " Abs Diff: " << diff_val
                          << " Abs Diff Threshold: " << static_cast<double>(abs_tol)
                          << " Rel->Abs Diff Threshold: "
                          << static_cast<double>(
                                 static_cast<double>(std::abs(C_ref_val)) * rel_tol)
                          << std::endl;
                status = false;
                return status;
            }
        }
    }
    return status;
}

template <typename T>
void random_initialize_matrix(T *A, size_t m, size_t n, size_t lda,
                              unsigned int seed = 0U) {
    std::default_random_engine eng(seed);
    // The best way to verify is to use integer values.
    std::uniform_int_distribution<int> dis(0, 5);
    // std::uniform_real_distribution<double> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    for (size_t j{0U}; j < n; ++j) {
        for (size_t i{0U}; i < m; ++i) {
            A[i + j * lda] = static_cast<T>(rand());
        }
    }
}

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

void generateNormalMatrixDouble(double *dA,long int m,long int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand()%3000;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormalDouble(gen, dA, m*n, 0, 1);
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
    const long int BLOCK_SIZE = 32;
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
    const long int BLOCK_SIZE = 32;
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

template <typename T>
__global__ void fixZeros(int m, int n, T *A, int lda) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n && A[i + j * lda] == 0) {
        A[i + j * lda] = 1;
    }
}


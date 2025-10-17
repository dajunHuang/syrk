#include <cublas_v2.h>
#include <cuda_fp16.h> // 引入半精度头文件
#include <cuda_runtime.h>
#include <curand.h> // 引入 cuRAND 头文件

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

// --- 错误检查宏 ---
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__);   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CURAND_CHECK(call)                                                    \
    do {                                                                      \
        curandStatus_t status = call;                                         \
        if (status != CURAND_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuRAND Error at %s:%d\n", __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())

#define NUM_WARPUP 5
#define NUM_REPEAT 20

// --- 新增部分：用于将 float 转换为 half 的 CUDA Kernel ---
__global__ void floatToHalfKernel(const float *in, __half *out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// --- 新增部分：在 GPU 上生成半精度随机矩阵 ---
void generateUniformMatrixHalf(curandGenerator_t &gen, __half *d_mat, int rows, int cols) {
    size_t num_elements = static_cast<size_t>(rows) * cols;

    // 1. 在 GPU 上创建一个临时的 float 缓冲区
    float *d_temp_float = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_float, sizeof(float) * num_elements));

    // 2. 使用 cuRAND 生成 float 类型的均匀分布随机数
    CURAND_CHECK(curandGenerateUniform(gen, d_temp_float, num_elements));

    // 3. 调用 Kernel 将 float 缓冲区转换为 half 缓冲区
    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);
    floatToHalfKernel<<<grid, block>>>(d_temp_float, d_mat, num_elements);
    CUDA_CHECK_LAST_ERROR();

    // 4. 释放临时缓冲区
    CUDA_CHECK(cudaFree(d_temp_float));
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    curandGenerator_t curandG = NULL; // cuRAND 生成器

    int m = 32768, n = 32768, k = 32768;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    // --- 修改点 1: 数据类型从 float* 改为 __half* ---
    __half *d_A = nullptr;
    __half *d_B = nullptr;
    __half *d_C = nullptr;

    // --- 修改点 2: alpha 和 beta 标量类型改为 __half ---
    __half one_h = 1.0f;
    __half zero_h = 0.0f;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // 创建并初始化 cuRAND 生成器
    CURAND_CHECK(curandCreateGenerator(&curandG, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(curandG, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandG, 1234ULL));

    int lda = m, ldb = k, ldc = m;

    // --- 修改点 3: 内存分配大小改为 sizeof(__half) ---
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(__half) * lda * k));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(__half) * ldb * n));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(__half) * ldc * n));

    // --- 修改点 4: 调用新的半精度随机数据生成函数 ---
    generateUniformMatrixHalf(curandG, d_A, lda, k);
    generateUniformMatrixHalf(curandG, d_B, ldb, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < NUM_WARPUP; ++i) {
        // --- 修改点 5: 调用 cublasHgemm ---
        CUBLAS_CHECK(cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                 &one_h, d_A, lda, d_B, ldb, &zero_h, d_C,
                                 ldc));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    for (int i = 0; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUBLAS_CHECK(cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                 &one_h, d_A, lda, d_B, ldb, &zero_h, d_C,
                                 ldc));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        time += temp_time;
    }
    time /= NUM_REPEAT;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[cublas hgemm] " << "m: " << m << ", n: " << n << ", k: " << k
              << ", "
              << "latency: " << time << " ms, "
              << "Effective TFLOPS: " << 2.0 * m * n * k / time / 1e9
              << " TFLOPS, " << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CURAND_CHECK(curandDestroyGenerator(curandG)); // 销毁 cuRAND 生成器
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
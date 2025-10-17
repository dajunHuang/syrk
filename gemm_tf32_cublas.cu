#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h> // 仍然使用 cuRAND 生成 float 数据

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

// --- 错误检查宏 (与之前相同) ---
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

#define NUM_WARMUP 5
#define NUM_REPEAT 20

// --- 使用 cuRAND 在 GPU 上直接生成 float 随机矩阵 ---
void generateUniformMatrixFloat(curandGenerator_t &gen, float *d_mat, int rows, int cols) {
    size_t num_elements = static_cast<size_t>(rows) * cols;
    // 直接使用 cuRAND 生成 float 类型的均匀分布随机数到目标设备指针
    CURAND_CHECK(curandGenerateUniform(gen, d_mat, num_elements));
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    curandGenerator_t curandG = NULL;

    int m = 32768, n = 32768, k = 32768;

    if (argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    // --- 数据类型仍然是 float* ---
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    float one = 1.0f, zero = 0.0f;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // --- 核心修改点: 允许 cuBLAS 使用 TF32 Tensor Core ---
    // 这是启用 TF32 加速的关键
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TF32_TENSOR_OP_MATH));

    // 创建并初始化 cuRAND 生成器
    CURAND_CHECK(curandCreateGenerator(&curandG, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(curandG, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandG, 1234ULL));

    int lda = m, ldb = k, ldc = m;

    // 内存分配大小为 sizeof(float)
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * lda * k));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * ldb * n));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * ldc * n));

    // 使用 cuRAND 生成随机 float 数据
    generateUniformMatrixFloat(curandG, d_A, lda, k);
    generateUniformMatrixFloat(curandG, d_B, ldb, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time = 0, temp_time = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < NUM_WARMUP; ++i) {
        // --- API 调用仍然是 cublasSgemm ---
        CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                 &one, d_A, lda, d_B, ldb, &zero, d_C,
                                 ldc));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    for (int i = 0; i < NUM_REPEAT; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));

        CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                 &one, d_A, lda, d_B, ldb, &zero, d_C,
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

    std::cout << "[cublas sgemm_tf32] " << "m: " << m << ", n: " << n << ", k: " << k
              << ", "
              << "latency: " << time << " ms, "
              << "Effective TFLOPS: " << 2.0 * m * n * k / time / 1e9
              << " TFLOPS, " << std::endl;

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CURAND_CHECK(curandDestroyGenerator(curandG));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
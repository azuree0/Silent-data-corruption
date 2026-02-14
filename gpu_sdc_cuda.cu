/**
 * NVIDIA CUDA GPU SDC test — deterministic XOR shift kernel.
 * Same algorithm as CPU; mismatch indicates possible GPU defect.
 */
#include "gpu_sdc_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void sdc_xor_kernel(unsigned int* out, unsigned int iters) {
    unsigned int x = 42u;
    for (unsigned int i = 0u; i < iters; i++) {
        x = x ^ (x << 13u);
        x = x ^ (x >> 17u);
        x = x ^ (x << 5u);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = x;
    }
}

extern "C" int gpu_sdc_cuda_run(int64_t iterations, int64_t* gpu_result) {
    if (!gpu_result) return 0;

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) return 0;

    unsigned int* d_out = nullptr;
    err = cudaMalloc(&d_out, sizeof(unsigned int));
    if (err != cudaSuccess) return 0;

    unsigned int iters_u = (iterations > 0xFFFFFFFF) ? 0xFFFFFFFFu : (unsigned int)iterations;
    sdc_xor_kernel<<<1, 1>>>(d_out, iters_u);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_out);
        return 0;
    }

    unsigned int h_out = 0;
    err = cudaMemcpy(&h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    if (err != cudaSuccess) return 0;

    *gpu_result = (int64_t)h_out;
    return 1;
}

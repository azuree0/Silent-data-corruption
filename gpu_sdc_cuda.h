/**
 * NVIDIA CUDA GPU SDC test — run XOR shift on GPU, compare with CPU reference.
 * Optional: only compiled when CUDA Toolkit is found.
 */
#ifndef GPU_SDC_CUDA_H
#define GPU_SDC_CUDA_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run XOR shift kernel on NVIDIA GPU. Returns 0 if CUDA unavailable or error.
 * On success: *gpu_result = GPU output, returns 1.
 */
int gpu_sdc_cuda_run(int64_t iterations, int64_t* gpu_result);

#ifdef __cplusplus
}
#endif

#endif

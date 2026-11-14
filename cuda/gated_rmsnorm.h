// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>

// Contiguous FP32 rows, 128 values per row (Qwen3.5-4B DeltaNet value head).
// y[r,c] = x[r,c] * rsqrt(mean(x[r,:]^2) + epsilon) * weight[c] * silu(gate[r,c]).
// Arrays must not alias, rows > 0, epsilon finite and positive; weights are the
// effective multiplicative norm weights, not an offset representation.
extern "C" cudaError_t qwen35_gated_rmsnorm_128(
    const float *x, const float *gate, const float *weight, float *y,
    int rows, float epsilon, cudaStream_t stream);

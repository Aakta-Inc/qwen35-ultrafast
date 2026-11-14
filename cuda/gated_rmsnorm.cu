// SPDX-License-Identifier: Apache-2.0
#include "gated_rmsnorm.h"
#include <cmath>

namespace {
// One warp per head: four contiguous values per thread, no shared-memory roundtrip.
__global__ void gated_rmsnorm_128(const float *__restrict__ x,
                                const float *__restrict__ gate,
                                const float *__restrict__ weight,
                                float *__restrict__ y, int rows, float epsilon) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= rows) return; // whole warp exits together
    const int lane = threadIdx.x;
    float v[4];
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = x[row * 128 + lane + i * 32];
        sum += v[i] * v[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    const float inv = rsqrtf(sum / 128.f + epsilon);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = lane + i * 32;
        const int idx = row * 128 + col;
        const float z = gate[idx];
        y[idx] = (v[i] * inv * weight[col]) * (z / (1.f + expf(-z)));
    }
}
}
extern "C" cudaError_t qwen35_gated_rmsnorm_128(
    const float *x, const float *gate, const float *weight, float *y,
    int rows, float epsilon, cudaStream_t stream) {
    if (!x || !gate || !weight || !y || rows <= 0 || rows > 16777215 ||
        !(epsilon > 0.f) || !std::isfinite(epsilon)) return cudaErrorInvalidValue;
    gated_rmsnorm_128<<<(rows + 3) / 4, dim3(32, 4), 0, stream>>>(x, gate, weight, y, rows, epsilon);
    return cudaGetLastError();
}

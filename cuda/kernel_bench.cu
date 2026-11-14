// SPDX-License-Identifier: Apache-2.0
#include "gated_rmsnorm.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { const cudaError_t error = (call); if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); std::exit(1); } } while (0)

// Explicit three-operation mathematical reference, not a proxy for llama.cpp.
__global__ void norm(const float *x, const float *w, float *y, int rows) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    float sum = 0;
    for (int i = lane; i < 128; i += 32) { const float v = x[row * 128 + i]; sum += v * v; }
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    const float inv = rsqrtf(sum / 128.f + 1e-6f);
    for (int i = lane; i < 128; i += 32) y[row * 128 + i] = x[row * 128 + i] * inv * w[i];
}
__global__ void silu(const float *g, float *y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = g[i] / (1.f + expf(-g[i]));
}
__global__ void multiply(const float *a, const float *b, float *y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

struct Device {
    float *p = nullptr;
    explicit Device(size_t n) { CHECK(cudaMalloc(&p, n * sizeof(float))); }
    ~Device() { cudaFree(p); }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
};

float measure(bool fused, int rows, float *x, float *gate, float *weights,
              float *tmp_norm, float *tmp_gate, float *out, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (fused) {
        CHECK(qwen35_gated_rmsnorm_128(x, gate, weights, out, rows, 1e-6f, stream));
    } else {
        norm<<<(rows + 3) / 4, dim3(32, 4), 0, stream>>>(x, weights, tmp_norm, rows);
        silu<<<(rows * 128 + 255) / 256, 256, 0, stream>>>(gate, tmp_gate, rows * 128);
        multiply<<<(rows * 128 + 255) / 256, 256, 0, stream>>>(tmp_norm, tmp_gate, out, rows * 128);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 50; ++i) CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start, stream));
    constexpr int repetitions = 1000;
    for (int i = 0; i < repetitions; ++i) CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaEventRecord(end, stream)); CHECK(cudaEventSynchronize(end));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, start, end));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
    CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph));
    return ms * 1000.f / repetitions;
}

double check_output(const std::vector<float> &x, const std::vector<float> &gate,
                    const std::vector<float> &weights, const float *device_out) {
    std::vector<float> out(x.size());
    CHECK(cudaMemcpy(out.data(), device_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double max_error = 0;
    for (size_t row = 0; row < x.size() / 128; ++row) {
        double sum = 0;
        for (size_t col = 0; col < 128; ++col) { double v = x[row * 128 + col]; sum += v * v; }
        double inverse = 1. / std::sqrt(sum / 128. + static_cast<double>(1e-6f));
        for (size_t col = 0; col < 128; ++col) {
            const size_t i = row * 128 + col;
            const double z = gate[i];
            const double expected = x[i] * inverse * weights[col] * z / (1. + std::exp(-z));
            const double error = std::abs(out[i] - expected);
            if (!std::isfinite(out[i]) || error > 2e-5 * (1. + std::abs(expected))) {
                std::fprintf(stderr, "numerical failure at %zu: %.9g versus %.9g\n", i, out[i], expected);
                std::exit(1);
            }
            max_error = std::max(max_error, error);
        }
    }
    return max_error;
}
int main() {
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    std::fprintf(stderr, "device=%s; synthetic FP32 norm+gate microbenchmark; NOT model speedup\n", prop.name);
    cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
    std::puts("rows,split_graph_us,fused_graph_us,split_over_fused,max_absolute_error");
    for (const int rows : {1, 7, 32, 256, 6144}) {
        const int n = rows * 128;
        std::vector<float> x(n), gate(n), weights(128);
        for (int i = 0; i < n; ++i) { x[i] = (i / 128 == 0) ? 0.f : std::sin(i * 0.17f); gate[i] = (i % 201 - 100) * 0.2f; }
        for (int i = 0; i < 128; ++i) weights[i] = 0.8f + i * 0.003f;
        Device dx(n), dg(n), dw(128), tmp(n), tmp_gate(n), out(n);
        CHECK(cudaMemcpy(dx.p,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dg.p,gate.data(),n*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dw.p,weights.data(),128*sizeof(float),cudaMemcpyHostToDevice));
        float split[3], fused[3]; double error = 0;
        for (int round = 0; round < 3; ++round) {
            // Reverse order to reduce systematic temperature/clock bias.
            for (int pass = 0; pass < 2; ++pass) {
                const bool is_fused = ((round + pass) % 2) != 0;
                float us = measure(is_fused,rows,dx.p,dg.p,dw.p,tmp.p,tmp_gate.p,out.p,stream);
                (is_fused ? fused : split)[round] = us;
                error = std::max(error,check_output(x,gate,weights,out.p));
            }
        }
        std::sort(split,split+3); std::sort(fused,fused+3);
        std::printf("%d,%.6f,%.6f,%.6f,%.9g\n",rows,split[1],fused[1],split[1]/fused[1],error);
    }
    CHECK(cudaStreamDestroy(stream));
}

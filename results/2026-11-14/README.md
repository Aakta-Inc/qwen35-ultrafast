# Initial RTX 5060 experiments

## Independent CUDA primitive

`kernel.csv` measures an original FP32 128-wide gated RMS normalization implementation against an explicit three-operation mathematical reference. Both use CUDA graphs. Each variant has 50 warmups and 1,000 timed graph replays per round; three rounds reverse order, and the median is reported. Inputs and all outputs are checked against independent double-precision CPU calculations. No Baseten implementation is used.

Hardware: NVIDIA GeForce RTX 5060 (8 GB), driver 580.178.04. Build: GCC 11.4.0, CUDA 12.8.93, CMake 3.31.6, release mode, `sm_120`. GPU power/clock configuration was not changed. Other CPU work and the native reference build were present, so this is an initial microbenchmark, not a final isolated performance result.

| 128-value rows | Three-operation graph | Fused graph | Ratio |
|---:|---:|---:|---:|
| 1 | 4.097 µs | 2.049 µs | 2.00× |
| 7 | 4.097 µs | 2.049 µs | 2.00× |
| 32 | 4.097 µs | 2.049 µs | 2.00× |
| 256 | 4.097 µs | 2.049 µs | 2.00× |
| 6,144 | 22.523 µs | 8.191 µs | 2.75× |

Maximum observed absolute numerical error: 7.30e-6. All elements passed `abs(error) <= 2e-5 * (1 + abs(reference))`. The one-row case is zero-input; the larger cases also include nonzero inputs and both positive/negative gates.

`kernel-memcheck.log` records NVIDIA Compute Sanitizer 12.8.93 memcheck with error exit code enabled: zero errors. Its instrumented timings are **not** performance evidence.

**Not a model speedup.** This primitive is not integrated into the model backend. llama.cpp already has its own operation fusions. A comparison to a decomposed mathematical reference does not establish a win over llama.cpp, SGLang or the application's existing runtime.

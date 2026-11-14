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

## Full native reference baseline and batch experiment

Completed with the Rust harness and pinned C++/CUDA llama.cpp backend, with no Python inference or benchmark process. Model: public Qwen3.5-4B Q4_K_M plus F16 vision projector, **not the private NF4/LoRA application model**. Backend revision: `9d925e7c02eb4e2e0da5be64a982cf79f9103fc7`. Compiler/toolkit as above; FFmpeg/FFprobe 7.0.2. Models, executable, native backend libraries and decoder binaries have hashes in each run's metadata.

One process per configuration; one excluded warmup then five timed repetitions per case; concurrency one; greedy output; thinking disabled; prompt cache disabled and response `cache_n=0` required. FlashAttention and CUDA graphs enabled. Context 16,384; logical batch 2,048; 6 CPU threads; all 33 model layers GPU-offloaded. The only candidate change is physical batch 512 → 1,024. Timing includes server preprocessing/media encoding/prompt evaluation and generation, plus local HTTP transport; it excludes model loading and client fixture encoding.

| Workload | Baseline mean | Batch-1024 mean | Baseline / candidate |
|---|---:|---:|---:|
| Text, 128 generated tokens | 1,189.89 ms | 1,191.41 ms | 0.999× |
| One 448×448 image | 124.91 ms | 125.02 ms | 0.999× |
| Eight independent images | 669.91 ms | 671.07 ms | 0.998× |
| 48 independent images | 3,996.19 ms | 4,008.57 ms | 0.997× |
| Native 12-second video | 1,009.07 ms | 1,010.98 ms | 0.998× |

Text decode throughput was 110.33 tokens/s for the baseline. This excludes prompt/media work and is not an end-to-end VLM throughput metric. Each visual request returned `YES` in both configurations; all five text outputs matched across configurations. Evaluated prompt/generated token counts matched case by case. This is a synthetic smoke check, not a model-quality evaluation or logit parity check.

**No demonstrated improvement from larger batches.** Differences below 0.4% do not establish a meaningful regression or improvement. No configuration passed the 1.15× target. There was no winning candidate warranting reverse-order confirmation. These are initial small-sample measurements, not confidence intervals or a throughput-under-load test.

### Native video sampling matters

The committed source video is 48 frames at 4 FPS. The pinned model-specific preprocessing overrides the generic requested FPS: the native decoder samples **24 frames at 2 FPS**, in temporal pairs. A separate debug run confirmed `fps=2.00`, `n_frames=48`, `target_frames=24` (see `video-diagnostic.txt`). Native video prompt evaluation was 2,473 tokens, versus 9,805 for 48 independent images. Do not interpret the approximately 4× latency difference as an equivalent-work optimization. Debug-run timings are excluded from the comparison.

### Excluded and diagnostic runs

- The initial native run completed its text/image cases but failed video probing with host FFmpeg 4.4.2. It has no completed summary and is excluded; a fresh full run used isolated FFmpeg 7.0.2.
- An exploratory matrix-dispatch probe set environment variables that this pinned backend only recognizes as **compile-time options**. It is excluded as an ineffective intervention; no dispatch speedup is claimed. The ineffective runner option was removed before publishing code.
- The debug run establishes GPU offload and video sampling, not performance.
- Kernel memory-check timings are instrumented and excluded from performance comparisons.

No system services, GPU clock policy, production weights or application code were changed. Model process memory was released on completion. The independent CUDA primitive is still not integrated into the native model baseline. The 15% whole-model target remains open.

### Reproduce the comparison

From the repository root:

```sh
cargo run --release --locked --bin compare -- \
  results/2026-11-14/baseline-512 results/2026-11-14/candidate-1024
```

For fresh runs, use the README command with the committed video fixture, then change only `--ubatch 1024` and the output directory/label. Use the same pinned build and toolchain. The exact build additionally set `CMAKE_CUDA_ARCHITECTURES=120`, `CMAKE_BUILD_TYPE=Release` and `GGML_CUDA_FA_ALL_QUANTS=OFF`. Upstream resolves the Blackwell CUDA compilation target to `sm_120a`. A future kernel integration must first establish that this primitive contributes enough latency to matter, preserve numerical/output behavior and then pass an independent, reversed-order end-to-end comparison.

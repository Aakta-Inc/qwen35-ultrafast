# Qwen3.5-4B Ultrafast

Independent Rust/CUDA inference research for Qwen3.5-4B on a single NVIDIA GPU.

**Status:** baseline and optimization experiments, not a completed custom inference engine. The Rust harness currently measures a pinned native C++/CUDA llama.cpp reference backend. No Baseten source is used or vendored. Our own execution code is Rust, CUDA C and limited C++; no Python runtime, bindings, model conversion or benchmark scripts.

## Goal

Reach at least **1.15× throughput** (or baseline latency / candidate latency >= 1.15) against a strong, same-hardware native baseline while preserving the defined quality contract. This is a research target, not a measured claim. A future SGLang comparison must be identified separately; beating llama.cpp does not establish beating SGLang.

The reason for a model-specific engine is rapid experiments with kernels, fusion, memory traffic and graph capture. Rust alone is not expected to make GPU-bound inference faster.

## Native baseline

The submodule pins llama.cpp to `9d925e7c02eb4e2e0da5be64a982cf79f9103fc7` (`b12381`, MIT). Only its native backend is used. Upstream has tooling in other languages, which is not required for this build or benchmark.

Build from source with a CUDA toolkit and CMake:

```sh
git submodule update --init --depth 1
cmake -S vendor/llama.cpp -B build/llama -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build build/llama --config Release --target llama-server -j 4
cargo build --release --locked
```

Alternatively use the upstream `b12381` CUDA 12.8 Linux binaries and matching runtime bundle. Set `LD_LIBRARY_PATH` to their extracted directories. The harness records the executable hash/version. Do not silently substitute a different build.

Obtain preconverted weights from [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), pinned revision `e87f176479d0855a907a41277aca2f8ee7a09523`: `Qwen3.5-4B-Q4_K_M.gguf` and `mmproj-F16.gguf`. Keep the pair and hashes fixed across a comparison. This quantized public model is **not** the private NF4/LoRA application model.

```sh
cargo run --release --locked -- \
  --server /path/to/llama-server \
  --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-F16.gguf \
  --out runs/baseline --label baseline --repeats 5
```

Create the parent `runs` directory first. Each output directory must be new. Run only on an available GPU; the harness does not stop other workloads. It starts its own loopback-only server and stops that child on completion or error.

The default workload is text generation plus 1, 8 and 48 synthetic 448×448 images. Images represent a red rectangle moving across a neutral background. **Image sequences are not native temporal video inputs.** They test compute scaling and basic vision correctness, not surveillance accuracy. Client PNG encoding is excluded from timing; server image processing and request/response latency are included. Model loading and one warmup per case are excluded. Cache reuse is disabled and verified from every response. Graphs and FlashAttention are enabled for the strong reference baseline.

Artifacts: metadata, raw warmup/timed responses, model/backend hashes, server log, mean/p50/p95 request latency, serial requests/minute and decoder throughput. GPU memory readings are snapshots, **not peak VRAM**. Small-sample p95 is descriptive only. Outputs are preserved for candidate comparisons. Do not infer end-to-end speed from decoder tokens/second.

## Optimization order

1. Measure the native baseline; isolate input size, prompt processing and decoding costs.
2. Tune batching with identical weights, inputs, graph policy and output budgets. Re-run the winner and baseline in reverse order.
3. Build independently implemented CUDA primitives for Qwen's dimensions, starting with profiled bottlenecks in normalization/gating, projections and Gated DeltaNet. Kernel-only gains are not model gains.
4. Integrate winning primitives into the independent runtime, with intermediate-state checks against the native reference.
5. Add native video semantics, tokenization/processor parity, adapter support and public quality evaluation before application integration.

No speedup is accepted from reducing frames/resolution, changing quantization, enabling warm prompt reuse, shortening answers or comparing unlike GPUs. Long-form generation and short scoring are separate workloads. Public fixtures/results must not contain customer footage, credentials or proprietary adapters.

## Development

```sh
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

Original code: Apache-2.0. Dependencies retain their licenses. See [design and provenance](docs/plans/2026-11-14-design.md).

## Independent CUDA experiment

`cuda/gated_rmsnorm.cu` implements FP32 RMS normalization plus SiLU gating for the 128-value DeltaNet heads. It is **not yet integrated into model inference**. The C API accepts effective norm weights (without any implicit +1 transformation).

```sh
cmake -S . -B build/kernels -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build/kernels -j 4
ctest --test-dir build/kernels --output-on-failure
build/kernels/qwen35-kernel-bench
```

The microbenchmark compares an explicit three-operation implementation with the fused implementation, **both using CUDA graphs**, and checks both against independent double-precision CPU calculations. It includes partial warps of rows, zero norm inputs, varied weights, and positive/negative gates. It does not compare with llama.cpp's own fusion, and its speedup must never be reported as a model speedup. Architecture `120` is specific to the first measured RTX 5060; choose the architecture for your own GPU.

Compare a batching candidate to its baseline:

```sh
cargo run --release --locked --bin compare -- runs/baseline runs/candidate
```

The comparator rejects changes in model/projector/executable identity, request hashes, evaluated token counts, graph policy and incomplete runs. It reports output agreement per sample. Hardware/thermal isolation must also be verified from the recorded metadata; agreement on these synthetic cases is not a quality benchmark.

For the additional native-video case, add `--video fixtures/moving-rectangle.mp4 --video-fps 4`. FFmpeg must be installed for the native backend's video decoder. This case uses `input_video` and is labeled separately from image sequences. See [fixture provenance](fixtures/README.md). Keep FPS and the fixture hash fixed across comparisons.

Native video requires an FFprobe build supporting the `stream_side_data` query used by the pinned backend. The initial host FFmpeg 4.4.2 failed this query; the measured runs use an isolated FFmpeg/FFprobe 7.0.2 build on `PATH`. The harness records both tool hashes/versions. No global upgrade is needed. For a CUDA toolkit outside standard system directories, expose its `lib`/`lib64` directory through `LD_LIBRARY_PATH` during linking and execution.

## First measured results

On the RTX 5060, the initial native reference averaged **125 ms for one image**, **3.996 s for 48 independent images**, and **1.009 s for the 12-second synthetic native-video case**. The video backend sampled 24 of the source's 48 frames; this is not equivalent to the 48-image workload. Increasing the physical processing batch from 512 to 1,024 showed no useful gain. The independent CUDA primitive achieved 2–2.75× versus its decomposed reference, but is not integrated into inference. **No whole-model speedup is established yet.**

[Raw measurements, checks and limitations](results/2026-11-14/README.md).

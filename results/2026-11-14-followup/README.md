# Profile-guided dispatch experiment — 2026-11-14

Test host: post tower, RTX 5060 8 GiB, driver 580.178.04. Same public Qwen3.5-4B Q4_K_M and F16 projector as the first baseline. No application model, service, GPU clocks or system installation was changed.

## Finding and hypothesis

Original C++ CUPTI tracing shows quantized matrix work dominates both decode and the 48-image case. The text snapshot's six leading quantized GEMV variants account for about 2.21 seconds of cumulative kernel time. The 48-image snapshot has multi-second aggregate quantized GEMM work, alongside vision GEMM, attention and recurrent processing. Normalization and SiLU gating are comparatively small, and the reference already fuses several of those operations. The standalone normalization microbenchmark therefore is not evidence of a large available model-level gain.

The user-supplied DeepSeek post motivates shape-specific matrix work and fused GPU execution. Its MoE routing and residual-stream implementations do not apply directly to this dense Qwen model. We tested the concrete hypothesis that larger dense Blackwell projections would run faster through dequantization plus cuBLAS than upstream quantized MMQ dispatch.

The [candidate patch](../../experiments/blackwell-prefill-cublas.patch) changes only Q4_K/Q5_K/Q6_K operations on compute capability 12.x with at least 128 columns and no experts. This is an experimental modification of the MIT reference backend, not an independently implemented model engine. No Baseten code was used.

## Uninstrumented results

Fresh reference process followed by fresh candidate process. One excluded warmup and five timed repetitions per workload. Graphs and FlashAttention enabled, batch 2048, microbatch 512, context 16384, six CPU threads, single request at a time. Identical requests, model hashes, media decoder hashes, evaluated prompt/generated token counts and output budgets. Native video retains the model's 24-frame sampling policy. Both binary directories were retained separately and native library resolution checked.

| Workload | Reference mean | Candidate mean | Reference / candidate |
|---|---:|---:|---:|
| Text, 128 generated tokens | 1188.64 ms | 1194.06 ms | 0.995× |
| One image | 124.31 ms | 149.92 ms | 0.829× |
| Eight independent images | 663.97 ms | 872.99 ms | 0.761× |
| 48 independent images | 3982.18 ms | 5236.73 ms | 0.760× |
| Native 12-second video | 1007.19 ms | 1316.72 ms | 0.765× |

**Rejected:** visual latency increased by approximately 21–32%. All paired synthetic outputs matched and caching remained disabled. This does not establish general numerical or task-quality parity. No winner warranted a reverse-order confirmation; these small samples are not confidence intervals. Extra dequantization and disabled MMQ fusion are plausible costs, not individually isolated causal conclusions.

`comparison.json` records the exact backend change: the CUDA library hash differs while the server executable and other native library hashes match. Raw metadata, responses and completion summaries are in `reference/` and `prefill-cublas/`. The comparison command requires explicit `--allow-backend-change`; without it, this pair is rejected.

## Diagnostic boundaries

The profile CSVs are separate instrumented runs, not these latency samples. They contain partial cumulative kernel time including loading and warmup; the final unflushed interval is excluded. Profiles report zero dropped records. The candidate trace confirms a real dispatch change: FP16 GEMM is its leading kernel (5.07 seconds cumulative), followed by Q4_K dequantization (2.12 seconds). CUPTI 13.0.85 was needed on this driver/GPU; CUPTI 12.8 flushing failed and its failed runs are excluded. See [profiler instructions](../../profiling/README.md).

The next promising target is preserving quantized math and existing fusion while specializing GEMV/GEMM work for the actual Qwen dimensions. Mixed text/vision batching also requires careful recurrent-state and positional-semantic checks; it cannot be enabled safely by merely changing a server flag. Speculative decoding requires an appropriate draft/verification implementation and acceptance measurements before it can be proposed as a gain.

The 1.15× whole-model target and independent inference runtime remain unfinished. This experiment narrows the search; it does not claim an improvement.

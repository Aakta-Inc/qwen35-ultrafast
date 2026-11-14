# Qwen quantized projection experiments — 2026-11-14

Post tower: RTX 5060 8 GiB, driver 580.178.04, CUDA 12.8. The pinned native reference, public Qwen3.5-4B Q4_K_M, F16 projector, input fixtures and inference settings are unchanged. These are original experimental modifications of the MIT llama.cpp reference, plus an original C++ test harness. They are not a completed independent model engine. No Baseten source was used.

## What changed

The decode candidates share activation loads between two output rows for the model's 2560-, 4096- and 9216-wide quantized projections. Four-, two- and one-warp layouts were tested. The conservative candidate retains the original four-warp summation order and excludes fused gating and the large vocabulary head. Prefill experiments separately changed column tiles to 64 or row tiles to 64. They retain the original Q4_K/Q5_K/Q6_K weight representation and activation quantization.

[Reproducible patches](../../experiments/README.md) remain separate from the default pinned submodule. They are experimental alternatives, not a stack to apply together.

## Numerical and memory checks

The [C++ projection probe](../../benchmarks/README.md) loads actual GGUF tensors, selecting every distinct quantized projection shape/type and the matching fused FFN up/gate pair. It tests zero and two signed nonzero activation inputs. The decode check covers 36 cases and **898,560 values**; the 196-column image-prompt check covers 33 cases and **30,105,600 values**. It tests representative tensors, not every layer or real-world task accuracy.

| Variant | Bit differences from reference | Maximum absolute error | Decision |
|---|---:|---:|---|
| Two rows, four warps | 0 | 0 | Fused FFN microbenchmark regresses |
| Two rows, two warps | 400,459 | 4.77e-7 | Deterministic full-model text differs; reject under current output contract |
| Two rows, one warp | 406,917 | 4.77e-7 | Projection experiment only; no accepted model gain |
| Non-gated two rows, four warps | 0 | 0 | Candidate for full-model comparison |
| Prefill column tile 64 | 4,101,506 | 2.38e-6 | Broad projection regressions; reject |
| Prefill row tile 64 | 4,908,960 | 3.34e-6 | Mixed small gains and larger regressions; reject |

All candidates passed the predeclared projection tolerance `abs(error)/(1+abs(reference)) <= 2e-5`. This tolerance was not relaxed to accept a candidate. Passing it does **not** override a different whole-model answer. The conservative candidate also passed NVIDIA Compute Sanitizer memcheck with **zero errors**, and its instrumented outputs still matched exactly. See `checks/` for logs and binary/dump hashes.

The comparison checker was exercised with both a truncated dump and a deliberately changed numerical value; both failed as intended. Four Rust tests, formatting, and Clippy passed. The C++ harness compiled with `-Wall -Wextra -Werror`.

## Projection timings do not establish a model speedup

For one nonzero activation input, the four-warp/two-row experiment reduced the attention-gate projection from 18.25 to 14.67 microseconds and the unfused FFN projection from 34.25 to 25.71 microseconds. However, fused FFN increased from 76.41 to 89.12 microseconds. This motivated excluding fused gating from the conservative candidate.

These are repeated, hot-weight graph probes with three rounds of 100 executions; CSVs retain every input case. Their memory working set differs from full model inference. Later prefill probes also ran while an unrelated CPU video-decoding job was active. Neither these ratios nor the fixed numerical tolerance establish the 1.15x end-to-end target.

## Contention and excluded measurements

An unrelated research job was observed running multiple FFmpeg decoders on the tower during full-model testing. It was left untouched. `checks/final-run-host.txt` records a snapshot with ten FFmpeg processes; each final run additionally records Linux load averages before loading and after each request. Load averages include runnable and uninterruptible tasks; they are not CPU utilization percentages.

`contended-reference/` and `contended-gemv/` contain the initial two-warp full suite. Its text answer changed, its visual answers matched, and its timings do not establish a gain. `unfused-quality/` is a two-repeat text/one-image smoke check for the conservative candidate, run during compilation and unrelated CPU work. Its requests and outputs match the reference, but **its timings are excluded**. CUDA sanitizer timings are also excluded. No background process or application service was stopped.

## Full-model comparison

The final run order is reference A, conservative candidate A, conservative candidate B, reference B. Each process uses one excluded warmup and five timed repetitions per case; context 16384, batch 2048, microbatch 512, six CPU threads, single request, graphs and FlashAttention enabled. This reverses order while retaining the same model, requests, media tools, frame sampling and output budgets. Native video still uses the model's 24-frame sampling policy, not 48 independent images.

All four runs completed. The request hashes, model/projector hashes, media decoder hashes, evaluated prompt/generated token counts and cache policy matched. **All paired synthetic outputs matched** for the conservative candidate in both orders.

| Workload | Reference / candidate, order A | Reference / candidate, reversed order B |
|---|---:|---:|
| Text, 128 generated tokens | 0.988x | 0.983x |
| One image | 0.993x | 0.991x |
| Eight independent images | 0.993x | 1.001x |
| 48 independent images | 0.993x | 1.003x |
| Native 12-second video | 0.994x | 0.987x |

**No demonstrated whole-model improvement; the 1.15x target is not met.** The small differences are not a reliable estimate of isolated performance under the observed CPU contention. They do not justify promoting the candidate. Full-model working-set behavior and host contention limit what can be inferred from the hot-weight projection gains; this experiment does not individually isolate those causes.

`comparison-a.json` and `comparison-b.json` contain exact means, library hashes and output checks. Their raw runs are in `reference-a/`, `unfused-a/`, `unfused-b/` and `reference-b/`. The first final reference also overlapped brief diagnostic file hashing/comparison; this is an additional reason not to treat these as isolated measurements. No CPU compilation ran during the four final suites.

The reference source and original backend binaries were restored after testing; the CUDA library hash matched the saved reference, and the GPU returned to 10 MiB / 0% utilization. Candidate binaries remain isolated for reproducibility. No candidate is enabled by default. The independent inference engine and target improvement remain unfinished; a quiet end-to-end benchmark window is needed before accepting any future gain.

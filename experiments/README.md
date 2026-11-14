# Controlled native backend experiments

These patches are experiments against the pinned MIT-licensed llama.cpp reference, not an independent inference engine implementation. No Baseten source is used. The default submodule stays unmodified; retain the original executable **and shared libraries** before rebuilding a candidate.

## Blackwell prefill cuBLAS dispatch

`blackwell-prefill-cublas.patch` routes dense Q4_K/Q5_K/Q6_K matrix operations with at least 128 columns on compute capability 12.x to dequantization plus cuBLAS. Smaller batches, decode and expert dispatch retain upstream behavior. It also prevents upstream MMQ-only fusion where that dispatch predicate is checked. The hypothesis is that larger matrix work may benefit from vendor GEMM enough to offset dequantization and lost fusion. This is a hardware/workload experiment, not a recommended default.

Apply from the project root:

```sh
git -C vendor/llama.cpp apply ../../experiments/blackwell-prefill-cublas.patch
```

Rebuild `llama-server` with the same build settings as the reference. Use separate library directories on `LD_LIBRARY_PATH` so each executable loads its matching libraries. Benchmark identical model files, prompts, frames, batch settings and sampling policy. For an explicitly changed backend:

```sh
cargo run --release --bin compare -- BASELINE_DIR CANDIDATE_DIR --allow-backend-change
```

This opt-in records both sets of executable/library hashes and still rejects model, input, evaluated-work, batch-policy and instrumentation mismatches. Synthetic output agreement is only a smoke check; it does not establish numerical or task-quality parity.

Restore the reference sources with `git -C vendor/llama.cpp apply -R ../../experiments/blackwell-prefill-cublas.patch` after the experiment. Rebuilding is required to restore the binary.

## Qwen quantized projection layouts

The `qwen-*-gemv.patch` experiments keep the GGUF weights and activation quantization fixed and specialize single-column dense Q4_K/Q5_K/Q6_K operations with Qwen input widths 2560, 4096 and 9216 on SM120. They do not alter vision preprocessing, prompt lengths or generation limits.

- `qwen-two-row-gemv.patch`: two rows per block, original four-warps summation. Projection outputs match exactly, but fused FFN microbenchmarks regress.
- `qwen-two-row-two-warp-gemv.patch`: two rows and two warps. Small numerical differences pass the projection tolerance, but the tested deterministic text completion changes; excluded under the current strict output agreement contract.
- `qwen-two-row-one-warp-gemv.patch`: two rows and one warp. Tested at projection level; not promoted as a model speedup.
- `qwen-unfused-two-row-gemv.patch`: keeps four warps and only uses two rows for non-gated projections up to 9216 output rows. The fused FFN and vocabulary head retain their reference paths.
- `qwen-prefill-tile64.patch`: narrower column tiles for dense Qwen prefill shapes. Slower on the measured image-prompt projections, so excluded.
- `qwen-prefill-row64.patch`: shorter row tiles with 128 threads for Blackwell Q4_K/Q5_K/Q6_K at column-tile width 112. This tests a different shared-memory/occupancy tradeoff. It is an experimental backend configuration, not a generally validated dispatch policy.

Apply **one patch at a time** to a clean pinned reference source tree, then rebuild. These are mutually exclusive experiments, not a patch stack. Retain separate binary/library directories and run the [projection probe](../benchmarks/README.md), then full-model comparisons. Results and acceptance decisions are recorded in [the GEMV experiment report](../results/2026-11-14-gemv/README.md). None of these patches changes the default submodule.

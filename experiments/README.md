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

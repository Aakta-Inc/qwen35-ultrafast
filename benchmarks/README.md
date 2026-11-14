# Quantized projection probe

`projection_probe.cpp` is an original C++ harness around the pinned native ggml API. It reads actual Q4_K/Q5_K/Q6_K weight tensors from the supplied GGUF; it does not generate substitute weights or alter quantization. It samples one tensor for every distinct `(type, input width, output rows, fused gating)` combination. Fused FFN probes use the matching up/gate pair. These are representative shapes, not every layer or an end-to-end quality evaluation.

Build against the pinned reference libraries, using your installation paths:

```sh
export LD_LIBRARY_PATH="$REFERENCE_LIB_DIR:$CUDA_ROOT/lib:$LD_LIBRARY_PATH"
c++ -std=c++17 -O2 -Wall -Wextra -Werror benchmarks/projection_probe.cpp \
  -I vendor/llama.cpp/ggml/include -L "$REFERENCE_LIB_DIR" \
  -lggml-cuda -lggml-base -lggml -o build/projection-probe
mkdir -p runs
build/projection-probe MODEL.gguf runs/projection-reference.bin > runs/projection-reference.csv
```

Run the same executable with the candidate library directory first in `LD_LIBRARY_PATH`, saving a new dump. Verify the loaded library hashes. Compare:

```sh
build/projection-probe --compare runs/projection-reference.bin runs/projection-candidate.bin
```

The checker requires matching tensor/shape/seed headers, finite values and completed dumps. It reports bit differences, maximum absolute error and maximum `abs(candidate-reference)/(1+abs(reference))`; it fails above `2e-5`. This fixed numerical tolerance is a projection regression check, not permission to change whole-model answers. Exact output agreement is checked separately by the Rust full-model comparator.

Each shape uses three deterministic activation vectors: zero plus two nonzero signed inputs. Timing excludes model loading and host/device input copies. There are ten warmups and three rounds of 100 repeated graph executions; the CSV records the median round mean. These hot-weight, repeated-graph timings include host submission and synchronization, and are not streaming whole-model latency. CPU contention can also affect them. Do not extrapolate their ratios into model speedups.

`--columns 196` probes image-prompt matrix batches; values from 1 to 512 are supported. The output/embedding head is excluded for multi-column probes because the measured model evaluates it only at selected logits positions. Keep the column count identical across a comparison. `--check` performs one execution per input and writes zero timing fields; use it for correctness and sanitizer runs, never performance.

For a batch-shape diagnostic, `--compare-first-column LEFT.bin RIGHT.bin` compares only the first output column while allowing different column counts. All other tensor/shape/seed contracts must match; both dumps must be complete and finite. The probe generates the same first activation column at every batch width. Use widths above one to retain the same case set. This isolates matrix batch sensitivity, not whole-model caption quality, and reports no speedup. The usual `--compare` still requires identical shapes and checks every output.

```sh
compute-sanitizer --tool memcheck --error-exitcode 86 \
  build/projection-probe MODEL.gguf runs/projection-checked.bin --check
```

Dumps use a versioned header, per-case contracts, raw native float values, and a completion marker. They are local diagnostics, not a portable model format. Keep model files and bulky dumps outside source control; publish hashes, numerical-check logs and CSV summaries instead. Missing/malformed files, unsupported models, GPU failures and incomplete dumps fail explicitly.

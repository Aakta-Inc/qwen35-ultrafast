# Packed Qwen visual prefill on an idle RTX 5060

## Intervention

The pinned server normally evaluates image embeddings and adjacent text separators in separate calls. The experimental patch packs complete image chunks and text into the existing prompt batch using the existing mixed-batch machinery. This reduces fragmented prompt evaluation. It keeps original Q4_K_M weights, F16 projector, vision processing, MRoPE position construction, sampler, graph policy, batch 2048 / microbatch 512, context and six CPU threads.

This is an independently developed patch against the MIT-licensed llama.cpp reference, not Baseten code and not a finished from-scratch runtime. See [the patch and reproduction notes](../../experiments/README.md#qwen-mixed-textimage-prefill).

## Controlled measurements

Each process waited for two consecutive checks with no GPU compute process, zero GPU utilization and at least 95% CPU idle in both one-second samples. No unrelated workload was stopped. The full order is reference A, packed A, packed B, reference B. Each case has one excluded warmup and five timed repeats. `idle.log` records the checks.

Only the server executable and server implementation library were replaced; the candidate uses the original reference GPU and other dependency libraries. Metadata records all hashes. Request hashes, model/projector hashes, generated/evaluated token counts and outputs are checked by the Rust comparator. Native video remains the same 12-second public synthetic fixture with 24 sampled frames; the 48-image case remains 48 separate images.

| Case | Reference A / packed A, ms | Speedup A | Reference B / packed B, ms | Speedup B |
| --- | ---: | ---: | ---: | ---: |
| Text, 128 generated tokens | 1189.35 / 1192.11 | 0.998x | 1191.46 / 1189.65 | 1.002x |
| 1 image | 124.37 / 107.83 | 1.153x | 124.97 / 108.24 | 1.155x |
| 8 images | 664.09 / 572.29 | 1.160x | 671.36 / 572.90 | 1.172x |
| 48 images | 3985.24 / 3365.90 | 1.184x | 4019.51 / 3380.09 | 1.189x |
| Native video | 1017.78 / 858.88 | 1.185x | 1011.40 / 866.70 | 1.167x |

**All four visual cases exceed 1.15x in both orders, with identical paired outputs and evaluated work.** Text remains unchanged within measurement variation. Speedup means reference latency divided by candidate latency: 1.15x corresponds to 15% higher serial throughput, not 15% lower latency. Across the visual measurements latency falls approximately 13–16%. The one-image result is close to the threshold and should not be treated as a universal guaranteed gain.

Timing is uninstrumented warm request wall time, including server media processing and excluding client fixture encoding, process startup and model loading. Five repeats per process and two orders are descriptive measurements, not a broad hardware/workload confidence bound. `comparison-a.json` and `comparison-b.json` are the comparator outputs. The extra quality probes are a separate diagnostic run and are not substituted for the original performance inputs.

## Scope

The separate `--quality-probes --images 1 --repeats 2` runs check five additional questions, each with an excluded warmup and two recorded answers. All paired answers and evaluated token counts match (`quality-comparison.json`): color `Red`, position `LEFT`, absent blue circle `NO`, motion `LEFT`, and the same one-sentence description. **Agreement includes baseline mistakes:** the generated eight frames move right (x increases from 24 to 73), so `LEFT` is wrong; the description calls the 64-by-80 rectangle a square and places it in the bottom-left corner. This is parity evidence, not an accuracy pass. No assertion was weakened to hide these errors.

`idle.log` contains the four performance-run checks followed by two transfer-profile checks; `quality-idle.log` contains the separate parity-run checks. Rust formatting, all four tests and strict Clippy checks pass. The server patch compiled on the tower; both run orders and the separate parity comparison completed successfully. The source patch applies cleanly to the pinned submodule. After testing, the local and remote submodules are clean and the tower's working server/GPU library hashes match the saved reference again.

The synthetic smoke suite cannot establish general VLM quality, surveillance accuracy, or exact internal numerical equivalence. Changed batching can change floating-point accumulation even when generated answers agree. The experiment remains a separately reproducible candidate; the default submodule and reference binaries are restored. No production service is changed.

The reference itself emits recurrent-memory non-consecutive-position warnings for image MRoPE positions; those warnings also appear in the candidate. They are not evidence of a new candidate-only failure. Both logs remain on the test tower for inspection.

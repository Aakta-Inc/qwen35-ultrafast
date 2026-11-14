# Matrix scheduling and caption diagnostics

Continuation of the [real-media experiment](../2026-11-14-real-media/README.md), using the same pinned model/projector, RTX 5060 and public pedestrian clip. Diagnostic runs request token probabilities and are deliberately excluded from performance comparison.

**Decision:** reject the full-tile matrix candidate. It is slower in the projection screen, does not fix the caption, and introduces an additional capitalization difference. The previous best candidate remains experimental; consistently greater than 15% improvement with preserved quality is still unproven. The default reference is unchanged. The second matrix candidate is prepared but **not built or tested** because the tower moved to another queued research job.

## Caption divergence

Both measured repeats reproduce the prior ordinary-run captions. After nine identical generated tokens (`A group of people are walking on a sidewalk`), the tenth token differs:

| Backend | log probability of ` in` | log probability of ` near` | Selected |
| --- | ---: | ---: | --- |
| Reference | -1.000003 | -1.024877 | ` in` |
| Packed, no checkpoints v2 | -1.037659 | -1.004902 | ` near` |

These values are identical across the two measured repeats for each backend. The reference's log-probability margin of 0.024874 for ` in` becomes a margin of 0.032757 for ` near` in the candidate. This localizes the wording branch to a close ranking flip under the same generated prefix; comparisons after that point have different text contexts and cannot isolate prefill drift. The diagnostic does not establish semantic quality equivalence or identify a single responsible kernel. Full responses and top-five alternatives are preserved in `caption-diagnostic.json` and both raw run directories. No timing from these instrumented runs is counted as a speed result.

## Batch-width sensitivity

The actual-weight projection probe uses identical first-column activations for widths 4, 432 and 512. It covers 33 cases, including three activation seeds, with 153,600 first-column values per comparison. All weights and activation generation are unchanged.

- **432 versus 512:** every compared value is bit-identical.
- **4 versus 512:** 101,090 values differ; maximum absolute difference 0.00375408 and maximum scaled difference 0.00277377. This exceeds the fixed `2e-5` projection regression threshold. This is a diagnostic comparison of two reference execution shapes, not an accepted candidate regression.

The native backend can dispatch small batches through MMVQ or few-row MMQ and larger batches through MMQ. Thus unchanged model files do not imply identical numerical execution when prompt grouping changes. These measurements isolate matrix sensitivity; they do not establish that matrix work alone causes the caption change. Attention, recurrent processing and accumulated state remain possible contributors.

The larger differences are in Q4_K/Q5_K cases; tested Q6_K cases differ by at most `5.96046e-7`. Source inspection gives a concrete possible mechanism: `ggml-cuda/mmq-load-tiles.cuh` precombines Q4_K block scales/minima into half-precision tile values, while the MMVQ implementation in `vecdotq.cuh` applies them in float arithmetic. MMQ also consumes stored half-precision activation sums. This is stronger evidence than attributing every difference to reduction order, but a controlled intermediate-state ablation is still needed to assign the caption change to a particular operation.

## Scheduling experiment

`experiments/qwen-full-tile-mmq.patch` tests complete output tiles per CUDA block for dense Qwen Q4_K/Q5_K/Q6_K prefill projections on SM120, at input widths 2560/4096/9216 and batch widths 32–512. It avoids split-reduction partial-result traffic and the fixup kernel where the reference would split work. Fused operations already use their separate full-tile path and are unchanged. This is an experiment against the pinned native reference, not independent inference-engine code.

The patch is not applied to the default submodule. Its performance and model-quality acceptance require separate ordinary benchmark runs after projection checks. Token-probability timings must not be used as speed evidence.

The full-tile build compiled successfully. Same-shape checks at 432 and 512 columns cover **144,998,400 values**: maximum absolute difference `3.57628e-6`, maximum scaled difference `1.05026e-6`, below the unchanged `2e-5` threshold. The candidate's first column remains bit-identical between those two batch widths. Passing this numerical screen is not a caption-quality pass.

The hot-weight projection screen is slower: ratio of summed case medians **0.95050x at 432 columns** and **0.95711x at 512 columns**, reference divided by candidate. Small key/value projections are particularly worse (approximately 0.67–0.69x at 432 columns). Each CSV preserves 33 cases, three rounds of 100 graph executions per case after ten warmups. These are probe timings, not model latency. The result is consistent with loss of useful parallelism outweighing the traffic savings; occupancy was not directly measured.

A second isolated patch, `qwen-prefill-two-blocks.patch`, tests 64-row/128-column tiles with 128 threads and up to two split-work blocks per SM for the selected dense Qwen dimensions. This couples a smaller resource footprint with more independently scheduled blocks, retaining split-work scheduling. It is an alternative to the full-tile patch, not a stack. The screen requires a greater-than-1% aggregate projection gain before proceeding to ordinary full-model runs; this screening threshold is not the 15% acceptance target.

That second experiment did not start: an earlier queued research job acquired the tower after the full-tile runs. Our pending waiter was removed without interrupting that job. No build, numerical, memory-safety or speed claim is made for the second patch. The proposed tile configuration also affects fused Q4_K/Q5_K/Q6_K operations at column-tile width 128; the extra split-work blocks apply only to the unfused dense Qwen dimensions. The actual selected tile width and occupancy must be checked when this experiment runs.

## Ordinary realistic-media checks

The full-tile CUDA library is combined with the previous packed/no-checkpoints v2 server, keeping all other libraries fixed. Candidate CUDA SHA-256: `a26a0c35212644b66dca34709f0df0cc115527ba579ed5d9156a8f30553ba0c4`. Run order is candidate A, reference A, reference B, candidate B. Each process covers the unchanged ten-case public corpus with one excluded warmup and three measured repeats per case. Media hashes, frame policy, prompts, output budgets, model files, evaluated token counts, decoder tools and cache policy match.

Both orders give **8/10 exact answer agreement**. The pedestrian caption remains the v2 caption (`near a grassy area ... white van`); fruit color changes from `orange` to `Orange`. The remaining eight answers match in every measured repeat. Capitalization alone does not demonstrate semantic quality loss, but the predeclared exact-output gate is not relaxed after seeing the result.

On the matching pedestrian-start short answer, the candidate achieves only **1.045x / 1.059x** versus the reference. This includes the existing prompt-packing optimization and is not an isolated gain from the new matrix patch. No >15% result is established for realistic video. `rejected-timing-screen-{a,b}.json` retain the same predeclared eight-case timing subset used previously, including explicit failed output agreement for fruit color. Its changed answer must not be counted as a speed win; no aggregate quality-preserving result is accepted. `quality-{a,b}.json` retain all ten cases.

## Controls and verification

The initial reference projection/caption diagnostics ran after CPU/GPU idle checks. Remaining GPU work used the existing shared FIFO GPU queue and repeated idle checks: no compute process, zero GPU utilization and at least 95% CPU idle in both one-second samples, checked twice. `matrix-idle.log` and `full-tile-idle.log` preserve observations. Other workloads were never stopped. No benchmark overlapped compilation.

Rust formatting, six tests and strict Clippy pass. The new tests cover diagnostic/performance separation, missing or malformed probability metadata, matching token streams and first-token divergence. The updated C++ probe compiles with `-Wall -Wextra -Werror` and runs both full-output and first-column comparisons on completed GPU dumps. The full-tile CUDA library builds and completes the projection and media runs. A separate CUDA memcheck was queued but did not start before the next job acquired the tower; no sanitizer pass is claimed.

The local and tower native sources are restored to the clean pinned revision. The tower working CUDA library is restored to reference SHA-256 `33f4b0048d15ae38ba02290e6122c2078b4573cfbeb7bb1c1e87c91bc65b6e97`. No test process from this experiment remains queued or running; another research job is using the tower. Isolated candidate binaries remain for reproduction. Public fixtures and exact output checks are limited evidence, not a general accuracy guarantee.

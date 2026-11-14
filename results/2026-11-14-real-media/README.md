# Real-media parity and checkpoint-free visual prefill

## Result and acceptance boundary

This experiment expands testing beyond synthetic rectangles to four public photographs and three real-video excerpts, with eight predeclared short-answer checks and two caption prompts. The faster candidate remains experimental: all eight short answers are correct and match the reference, and the fruit caption matches, but the pedestrian caption varies. **Exact answer parity is 9/10, not a general same-quality guarantee.** No production service or default reference is replaced.

The new optimization combines the earlier mixed text/image prompt batching with removal of checkpoint-related prompt-tail splits for eligible uncached Qwen visual requests. The corrected patch requires an actual image chunk, not merely a media-capable tokenizer. Source: [`qwen-packed-no-checkpoints.patch`](../../experiments/qwen-packed-no-checkpoints.patch). The initial version is retained as `qwen-packed-no-checkpoints-initial.patch` for reproducibility and is rejected because it also affected text-only requests.

The corrected candidate retains **1.19–1.40x** on the four original synthetic visual workloads in both run orders, with all answers matching, including the unchanged text-only case. On the realistic short-answer set, gains vary substantially: five image checks reach approximately 1.15–1.30x, the building check 1.11–1.14x, pedestrian video 1.06–1.08x, and the low-frame-count tree video 1.32–1.36x. **This is not a consistent >15% improvement across realistic workloads, nor a passed general-quality gate.**

### Corrected candidate: realistic short-answer cases

| Case | Reference / candidate, E (ms) | Speedup E | Speedup G, reverse order |
| --- | ---: | ---: | ---: |
| People count | 167.08 / 130.02 | 1.285x | 1.301x |
| Indoor setting | 184.02 / 145.90 | 1.261x | 1.255x |
| Fruit color | 137.38 / 106.89 | 1.285x | 1.293x |
| Building setting | 265.49 / 232.03 | 1.144x | 1.112x |
| Sudoku heading | 188.89 / 155.37 | 1.216x | 1.150x |
| Sudoku digit | 181.59 / 142.35 | 1.276x | 1.179x |
| Pedestrians, 16 sampled frames | 1475.78 / 1362.26 | 1.083x | 1.057x |
| Tree, 4 sampled frames | 162.35 / 122.67 | 1.323x | 1.357x |

E compares `real-reference-c` with `real-v2-e`; G compares `real-v2-g` followed by the fresh `real-reference-g`. Both use three measured repeats after one warmup. The aggregate must be calculated as the ratio of summed request times, not the average of case speedups. E is **1.152x** and reverse-order G is **1.126x**. The larger video dominates the aggregate; the whole eight-case mix does not reliably clear 1.15x.

### Corrected candidate: unchanged original synthetic suite

| Case | Speedup E | Speedup F, reverse order |
| --- | ---: | ---: |
| Text, 128 generated tokens | 0.999x | 1.000x |
| 1 image | 1.402x | 1.395x |
| 8 images | 1.224x | 1.237x |
| 48 images | 1.195x | 1.194x |
| Native 12-second video | 1.206x | 1.198x |

All five cases have matching outputs and evaluated work in both corrected-candidate comparisons. Five timed repeats per process, one excluded warmup per case. Speedup is reference latency / candidate latency, equivalent to serial throughput improvement; it is not the percentage latency reduction.

## Quality evidence

The [frozen corpus](../../fixtures/real-media/corpus.json) contains prompts, budgets, exact file hashes and independently written short-answer expectations. Labels are never passed to the model. Sources and preparation are documented [with the fixtures](../../fixtures/real-media/README.md). Only public OpenCV samples are used; the media itself is not redistributed here.

Short-answer labels: two people, indoors, orange citrus wedge, camera outside the building, Sudoku heading, first top-row digit 6, pedestrians walking outdoors, and green tree leaves. `short-answer-review.json` records the corrected candidate's 8/8 result across three repeats per case. The reverse-order output comparison also matches on all eight cases. Repeats test stability, not eight additional independent examples per repeat.

The reference pedestrian caption mentions a building, parking lot and grass. The earlier packed candidate omits the parking lot and grass. The checkpoint-free candidate mentions grass, the building and a parked white van. Inspection of the later clip at 0, 4 and 7 seconds supports the described visible scene, but these are different captions and do not satisfy exact-output parity. The matching fruit caption includes a mango identification that is not independently confirmed. No broad accuracy pass is claimed from matching a baseline caption.

The diagnostic comparator's `--quality-only` mode reports output and generated-length differences without speed claims. The normal performance comparator still rejects changed evaluated work. The `--cases` selection records an explicit case subset; it never silently drops mismatching cases. The speed subset here is all eight closed-answer cases, excluding both captions consistently. Raw full-suite results remain available.

## Bottleneck and intervention

The separate packed-candidate CUPTI snapshot reports zero dropped records. Partial cumulative kernel time is about 11.31 seconds, including initialization and warmup: quantized GEMM accounts for 3.99 seconds, CUTLASS GEMM 1.08 seconds, attention 0.93 seconds, and Gated DeltaNet 0.70 seconds. These are overlapping diagnostic activity totals, not per-request latency or a normalized model breakdown. Matrix work remains the largest target.

Source inspection also found avoidable small prompt-tail evaluations created to support recurrent-state checkpoints, even for requests that disable prompt reuse. The candidate disables those splits and snapshots only for single-slot, uncached, non-speculative Qwen image completion without adapters or shared prefixes. All other paths retain the reference behavior. No pixels, sampled frames, weights, quantization, output budgets, graph policy or FlashAttention settings are reduced.

Increasing the physical processing batch from 512 to 1024 on the initial checkpoint-free candidate yielded only 1.000–1.025x on the eight closed-answer cases, with essentially no pedestrian-video gain. Outputs remained equal to that candidate. It is not promoted; the final candidate keeps batch 2048 and physical batch 512.

A further `qwen-packed-boundary.patch` experiment lets an image at a batch boundary enter the regular prompt batch instead of a separate decoder call. It compiled and completed the full real-media suite, but ratios versus the corrected candidate were only 0.992–1.006x, including 0.992x on pedestrian video. It is retained as a rejected experiment, not part of the selected patch.

## Controls and run inventory

Hardware: post tower RTX 5060, same public Q4_K_M model and F16 projector, six CPU threads, one request at a time. Model, library, media-tool and input hashes are in each run's metadata. Candidate directories replace only the server implementation; GPU and other dependency libraries match the reference. Timing uses warm HTTP wall time, with one excluded warmup per case. Real-media runs use two repeats for initial screening and three for confirmation; synthetic confirmations use five.

Each process waits for two consecutive idle checks: no GPU compute process, 0% GPU utilization and at least 95% CPU idle in both one-second samples. `idle.log` records the run sequence. Unrelated workloads are not stopped. These small samples describe this machine and corpus; they are not statistical guarantees for arbitrary workloads.

- `reference-a`, `packed-a`: initial two-repeat real-media screen. Their legacy `input` description still says synthetic; the embedded corpus and request hashes identify the actual public-media inputs. Subsequent metadata uses the corrected description.
- `profile`: separate instrumented real-media run, excluded from latency comparisons; it also has the legacy input-description label while recording the correct corpus.
- `no-checkpoints-b`, `packed-b`, `reference-b`: initial candidate / previous candidate / original reference, three repeats.
- `real-ubatch1024-c`, `real-reference-c`, `real-no-checkpoints-c`: batch ablation and reverse-order three-repeat checks of the initial candidate.
- `synthetic-*-c`, `synthetic-*-d`: initial candidate's five-repeat synthetic runs in both orders. They exposed the text-only regression and are not accepted replacement results.
- `real-v2-e`, `synthetic-v2-e`, `synthetic-v2-f`, `synthetic-reference-f`: corrected candidate validation. `synthetic-reference-d` precedes the corrected E run; corrected F precedes the fresh F reference.
- `real-v2-g`, `real-reference-g`: reverse-order corrected-candidate validation on the full real-media suite.
- `real-boundary-h`: rejected additional batch-boundary experiment, compared with `real-v2-g`.

The pedestrian excerpts retain 768×576 source resolution and 16 sampled frames (eight temporal chunks). The low-resolution tree video is different: it has 19 encoded frames over 7.87 seconds but a nominal rate near 15 FPS. The pinned backend uses frame count / nominal FPS, clamps to four sampled frames, and produces two temporal chunks. Both variants use that same behavior. Its larger speedup must not be presented as representative of a full eight-second, 16-frame video workload. Correct handling of this variable-rate source is a separate baseline limitation, not an optimization introduced here.

## Verification and limitations

The text regression revealed that `server_tokens.has_mtmd` is a tokenizer capability flag, not evidence that a request contains an image. The corrected candidate looks up the actual first media chunk and requires an image before changing checkpoint behavior. The 128-token text case provides an end-to-end regression check with the same request hash and seed.

Rust formatting, six tests and strict Clippy pass. Tests cover corpus hash tampering, changed model/settings rejection, and the rule that differing-length answers cannot become speed claims. Both candidate server versions compile on the tower. No CUDA kernel is changed in this experiment. Public fixtures are a diagnostic corpus, not a replacement for independent task-level accuracy evaluation. Caption divergence and variable-rate video sampling remain material limitations.

After the final screen, both local and tower native submodules are clean. The tower's working CUDA and server-implementation library hashes match the preserved reference (`33f4b004...` and `58c966c5...` respectively), and no GPU compute process remains from these tests. Isolated candidate binaries remain available for reproduction. The final Rust harness also avoids constructing unused synthetic inputs during corpus setup; this affects untimed setup only, not request bodies or measured inference work.

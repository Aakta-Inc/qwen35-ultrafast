# Idle-tower sampling experiment

RTX 5060, unchanged Q4_K_M model/projector and pinned native reference. The candidate moves the existing greedy sampler to the native backend (`--backend-sampling`); request bodies, frame sampling, precision, batch sizes and output budgets stay unchanged. No Baseten code is used.

After unrelated CPU work finished, each process waited for two consecutive idle checks: no GPU compute process, 0% GPU utilization, and at least 95% CPU idle in both one-second intervals. `idle.log` records the checks. Order: reference A, GPU A, GPU B, reference B. Each process uses one excluded warmup and five measured repeats per case.

| Case | Reference / candidate, A | Reference / candidate, B |
| --- | ---: | ---: |
| Text, 128 generated tokens | 1.008x | 1.010x |
| 1 image | 0.976x | 0.982x |
| 8 images | 0.994x | 0.999x |
| 48 images | 1.008x | 1.002x |
| Native video | 1.007x | 0.994x |

All paired outputs agree, with matching request hashes, evaluated work and model/library identities. Native video still samples 24 frames. These synthetic outputs do not establish general task-quality parity. **No case reaches 1.15x; CPU sampling remains the default.** The approximately 1% text gain does not justify promoting this as the requested improvement.

`comparison-a.json` and `comparison-b.json` were generated with the Rust comparator's `--allow-sampling-change` option. That option only permits sampler placement to differ; it still requires identical backend binaries/libraries and batch policy. Raw metadata, responses and completion summaries are preserved for all four runs. Profiling is excluded from these measurements.

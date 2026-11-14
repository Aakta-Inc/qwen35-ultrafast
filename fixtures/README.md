# Public synthetic video fixture

`moving-rectangle.mp4` is an original synthetic fixture: a 448×448 white background with a 64×80 red rectangle moving right, 4 FPS, 12 seconds, 48 frames, no audio. It contains no real footage. SHA-256: `855887d37e9d837d64f879f486fa9dfe5ee6f8f7df6e08400c2bd02937c6ecc7`.

Created using the native FFmpeg executable (no Python):

```sh
ffmpeg -nostdin -v error \
  -f lavfi -i color=c=white:s=448x448:r=4:d=12 \
  -f lavfi -i color=c=red:s=64x80:r=4:d=12 \
  -filter_complex '[0:v][1:v]overlay=x=24+7*n:y=160:shortest=1' \
  -c:v libx264 -threads 1 -pix_fmt yuv420p moving-rectangle.mp4
```

Use the committed binary for exact comparisons: encoder versions can change encoded bytes. The model receives `input_video`, not a list of independent image attachments. The harness explicitly sets video FPS (default 4) and records the file hash. Server logs must confirm the frames actually processed; the requested rate alone is not proof of the processed count. This fixture tests basic media handling and compute cost, not real-world reasoning or action recognition.

Observed on the pinned Qwen backend: its model-specific policy overrides `--video-fps 4` and samples **24 frames at 2 FPS**, grouped into temporal pairs. This was confirmed in a debug run, not inferred solely from the requested setting. See `results/2026-11-14/video-diagnostic.txt`. Keep this effective sampling policy unchanged in a performance comparison.

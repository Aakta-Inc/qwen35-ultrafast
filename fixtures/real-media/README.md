# Public photographic and video parity set

`corpus.json` fixes ten prompts and output budgets over four photographs and three video excerpts. Eight short-answer labels were written from visual inspection before inference. Two captions are reviewed separately; exact answer equality is not the same as accuracy. These ten cases are a small diagnostic set, not a general VLM benchmark or a sealed evaluation set.

Media comes from the OpenCV sample-data tree at commit `59b6aad64b89f15247542d0f0418d4a7477a8f91`:
https://github.com/opencv/opencv/tree/59b6aad64b89f15247542d0f0418d4a7477a8f91/samples/data

The source repository includes its own license and notices. Media is downloaded separately, not redistributed in this repository. No customer footage, identities, private adapters or private model weights are used.

Photographs: `basketball1.png`, `building.jpg`, `fruits.jpg`, `sudoku.png`. Video sources: `vtest.avi` (outdoor pedestrians) and `tree.avi` (tree foliage). Preserve original image dimensions. Derive the video fixtures with FFmpeg:

```sh
ffmpeg -i vtest.avi -t 8 -an -c:v libx264 -crf 18 -pix_fmt yuv420p pedestrians-start.mp4
ffmpeg -ss 30 -i vtest.avi -t 8 -an -c:v libx264 -crf 18 -pix_fmt yuv420p pedestrians-later.mp4
ffmpeg -i tree.avi -t 8 -an -c:v libx264 -crf 18 -pix_fmt yuv420p tree.mp4
```

The two pedestrian excerpts are 768×576 at 10 FPS and eight seconds long. The tree excerpt is 320×240, approximately 15 FPS and 7.87 seconds. Encoding versions can change the output bytes; the recorded manifest hashes identify the exact measured files. For a newly generated corpus, record its new hashes and run both reference and candidate against that same frozen corpus. Never reuse reference results from different media hashes. Put media beside the manifest, or adjust its relative file paths before freezing it.

Run the Rust harness with `--corpus PATH/TO/corpus.json`; this replaces its synthetic cases. The runner validates every media hash before starting inference and records the manifest in metadata. All resizing, video sampling and inference remain the pinned native backend's responsibility, with unchanged policy across candidates.

For answer comparison use `compare BASELINE CANDIDATE --allow-backend-change --quality-only`. This reports differing outputs and generated lengths without speed claims. Ordinary performance comparisons continue to reject changed evaluated work. Each manifest case contains a review label or an expected short answer; labels are not supplied to the model.

Measured video encoding used FFmpeg 7.1.5 / Apple clang 21.0.0. The tree output is variable-rate: 19 encoded frames over 7.87 seconds despite its approximately 15 FPS nominal rate. The pinned backend samples only four frames from it using its frame-count/rate calculation. Treat this as a low-frame-count case, not as a 16-frame equivalent of the pedestrian clips. Source SHA-256: `vtest.avi` = `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`; `tree.avi` = `4666099d0f704e310047b2f0a5ec9f936cb76a7271de9a2e70a0c57f82ac82dc`.

# GPU kernel diagnostics

`cupti_trace.cpp` is an original C++ preload library for Linux. It writes sorted, cumulative kernel-time snapshots from CUPTI. These include loading and warmup, can overlap in time, and omit the final unflushed records when the runner stops the server. **They are not request latency or performance comparisons.** Inspect `dropped_records` before interpreting a snapshot.

Tested on RTX 5060 / driver 580.178.04 with CUPTI **13.0.85** and CUDA 12.8 headers/runtime. CUPTI 12.8 failed activity flushing with `CUPTI_ERROR_INVALID_DEVICE` on this setup. Do not substitute its ABI: this source uses `CUpti_ActivityKernel10`.

With toolkit locations supplied by your installation:

```sh
c++ -std=c++17 -O2 -shared -fPIC profiling/cupti_trace.cpp \
  -I "$CUPTI_ROOT/include" -I "$CUDA_ROOT/include" \
  -L "$CUPTI_ROOT/lib" -lcupti -pthread -o build/libqwen-profile.so
```

Put the matching CUPTI library directory on `LD_LIBRARY_PATH`, then add `--profile-library build/libqwen-profile.so` to the normal Rust benchmark command. The runner sets the target executable and output path, marks metadata as instrumented, and saves `kernels.csv` in the run directory. The preload ignores decoder child processes. The comparison command rejects instrumented runs even with `--allow-backend-change`.

Run uninstrumented measurements separately. A fast kernel alone does not establish a whole-model improvement.

/*
 * Profile-driven multi-GPU prediction and queueing comparison harness.
 */
#ifndef BENCHMARK_H
#define BENCHMARK_H

typedef struct {
    const char *name;
    double service_slope;
    double service_intercept;
    double memory_slope;
    double memory_intercept;
    int max_batch_size;
} GpuProfile;

int run_gpu_benchmarks(void);

#endif

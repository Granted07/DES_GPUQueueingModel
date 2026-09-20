/*
 * Placeholder-profile GPU benchmark predictions and CoV queueing experiment.
 * Values are model inputs, not live hardware measurements.
 */
#include "benchmark.h"

#include "sim_core.h"

#include <stdio.h>

#define BENCHMARK_REPETITIONS 5
#define BENCHMARK_ARRIVALS 10000
#define BENCHMARK_WARMUP 1000
#define BENCHMARK_LAMBDA 0.02
#define BENCHMARK_A 4
#define BENCHMARK_QUEUE_CAPACITY 1024

static const GpuProfile profiles[] = {
    /* PLACEHOLDER VALUES TO BE REPLACED WITH REAL PHASE 1 PROFILING DATA. */
    {"RTX_2080_placeholder", 2.0, 5.0, 11.0, 70.0, 32},
    {"Titan_Xp_placeholder", 2.5, 6.0, 12.0, 80.0, 32},
    {"T4_placeholder", 1.8, 7.0, 10.0, 65.0, 32}
};

static int write_profile_predictions(void)
{
    FILE *output = fopen("gpu_benchmark_predictions.csv", "w");
    size_t profile_index;
    int batch_size;

    if (output == NULL) {
        perror("Unable to open GPU prediction CSV");
        return 1;
    }
    fprintf(output, "gpu_name,batch_size,predicted_latency_ms,"
                    "predicted_throughput_img_s,predicted_memory_mb\n");
    for (profile_index = 0;
         profile_index < sizeof(profiles) / sizeof(profiles[0]);
         profile_index++) {
        const GpuProfile *profile = &profiles[profile_index];
        for (batch_size = 1; batch_size <= profile->max_batch_size;
             batch_size *= 2) {
            double latency = profile->service_slope * (double)batch_size +
                             profile->service_intercept;
            double throughput = (double)batch_size / (latency / 1000.0);
            double memory = profile->memory_slope * (double)batch_size +
                            profile->memory_intercept;
            fprintf(output, "%s,%d,%.6f,%.6f,%.6f\n", profile->name,
                    batch_size, latency, throughput, memory);
        }
    }
    fclose(output);
    return 0;
}

static int write_cov_comparison(void)
{
    const double cov_values[] = {0.0, 0.5, 1.4};
    FILE *output = fopen("gpu_benchmark_cov_comparison.csv", "w");
    size_t profile_index;
    size_t cov_index;
    int batch_size;

    if (output == NULL) {
        perror("Unable to open GPU CoV CSV");
        return 1;
    }
    fprintf(output, "gpu_name,batch_size,cov,mean_E_W,mean_E_L,"
                    "mean_P_block\n");
    for (profile_index = 0;
         profile_index < sizeof(profiles) / sizeof(profiles[0]);
         profile_index++) {
        const GpuProfile *profile = &profiles[profile_index];
        for (batch_size = 1; batch_size <= profile->max_batch_size;
             batch_size *= 2) {
            for (cov_index = 0; cov_index < 3; cov_index++) {
                double sum_wait = 0.0;
                double sum_queue = 0.0;
                double sum_block = 0.0;
                int repetition;
                for (repetition = 0; repetition < BENCHMARK_REPETITIONS;
                     repetition++) {
                    SimulationConfig config = {
                        BENCHMARK_LAMBDA, BENCHMARK_A, batch_size,
                        profile->service_slope, profile->service_intercept,
                        BENCHMARK_QUEUE_CAPACITY,
                        BENCHMARK_ARRIVALS + BENCHMARK_WARMUP,
                        (unsigned int)(700000 + profile_index * 10000 +
                                       batch_size * 100 + cov_index * 10 +
                                       repetition),
                        0.0, 0.5, 0, BENCHMARK_WARMUP, cov_values[cov_index]
                    };
                    SimulationResults results = simulate(&config);
                    sum_wait += results.expected_wait_miss_only;
                    sum_queue += results.expected_number_in_queue;
                    sum_block += results.blocking_probability;
                }
                fprintf(output, "%s,%d,%.1f,%.10f,%.10f,%.10f\n",
                        profile->name, batch_size, cov_values[cov_index],
                        sum_wait / BENCHMARK_REPETITIONS,
                        sum_queue / BENCHMARK_REPETITIONS,
                        sum_block / BENCHMARK_REPETITIONS);
            }
        }
    }
    fclose(output);
    return 0;
}

int run_gpu_benchmarks(void)
{
    puts("WARNING: GPU benchmark outputs use PLACEHOLDER profiles pending "
         "real Phase 1 profiling data; they are model predictions, not "
         "hardware measurements.");
    if (write_profile_predictions() != 0) {
        return 1;
    }
    return write_cov_comparison();
}

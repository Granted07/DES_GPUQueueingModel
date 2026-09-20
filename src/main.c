#include "sweep.h"
#include "sim_core.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int run_profile_batch(int argc, char **argv)
{
    SimulationConfig config = {0};
    SimulationResults results;
    if ((argc != 7 && argc != 9) ||
        strcmp(argv[1], "--profile-batch") != 0 ||
        strcmp(argv[3], "--service-mean") != 0 ||
        strcmp(argv[5], "--service-cov") != 0) {
        fprintf(stderr, "Invalid profile-batch arguments\n");
        return 1;
    }
    config.lambda = 0.05;
    config.a = 4;
    config.b = atoi(argv[2]);
    config.service_slope = 0.0;
    config.service_intercept = atof(argv[4]);
    config.queue_capacity = MAX_QUEUE_CAPACITY;
    config.arrivals_to_generate = 105000;
    config.seed = argc == 9 ? (unsigned int)strtoul(argv[8], NULL, 10)
                            : 424242U;
    if (argc == 9 && strcmp(argv[7], "--seed") != 0) {
        fprintf(stderr, "Invalid profile-batch seed arguments\n");
        return 1;
    }
    config.p_hit = 0.0;
    config.cache_lookup_latency = 0.5;
    config.enable_cache_sim = 0;
    config.warmup_arrivals = 5000;
    config.service_time_cov = atof(argv[6]);
    results = simulate(&config);
    printf("RESULT,%.10f,%.10f,%.10f\n",
           results.expected_wait_miss_only,
           results.expected_number_in_queue,
           results.blocking_probability);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--profile-batch") == 0) {
        return run_profile_batch(argc, argv);
    }
    if (run_sweep("queue_cache_results.csv") != 0) {
        return 1;
    }
    printf("Wrote queue_cache_results.csv\n");
    return 0;
}
